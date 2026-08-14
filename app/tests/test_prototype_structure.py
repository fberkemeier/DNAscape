"""Structural regression tests that do not require importing NiceGUI."""

from __future__ import annotations

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent

PAGE_MODULES = (
    "overview.py",
    "data_management.py",
    "mapping.py",
    "simulation.py",
    "plotting.py",
    "cached_files.py",
    "resources.py",
)

PAGE_LABELS = (
    "Overview",
    "Data Manager",
    "Mapping",
    "Simulation",
    "Plotting",
    "Cached Files",
    "Resources",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_all_seven_page_modules_exist_and_define_render() -> None:
    for file_name in PAGE_MODULES:
        path = APP_DIR / "pages" / file_name
        assert path.is_file(), file_name
        definitions = {
            node.name
            for node in _tree(path).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "render" in definitions, file_name


def test_prototype_never_imports_scientific_package() -> None:
    for path in APP_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                assert all(alias.name != "dnascape" and not alias.name.startswith("dnascape.") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "dnascape" and not str(node.module).startswith("dnascape.")


def test_shell_exposes_complete_navigation_and_local_entrypoint() -> None:
    source = (APP_DIR / "main.py").read_text(encoding="utf-8")
    for label in PAGE_LABELS:
        assert f'"{label}"' in source
    assert "127.0.0.1" in source
    assert "dnascape_logo_small.svg" in source
    assert 'mark("nav-logo-overview")' in source
    assert 'mark("navigation-drawer")' in source
    assert 'mark("drawer-toggle")' in source
    assert "value=True" in source
    assert "top_corner=False" in source
    assert 'aria-label="Collapse or expand navigation"' in source
    assert 'aria-label="Go to Overview"' in source
    assert "GITHUB_ICON_SVG" in source
    assert 'ui.icon("code")' not in source
    assert '("data-management", "Data Manager", DATA_ICON)' in source
    assert '"table_view"' not in source
    assert '"database"' not in source
    assert (APP_DIR / "__main__.py").is_file()


def test_workflow_copy_and_data_sections_use_updated_ui_semantics() -> None:
    ui_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (APP_DIR / "pages").glob("*.py")
    }
    assert "Data Management" not in "\n".join(ui_sources.values())
    assert '"Data Manager"' in ui_sources["overview.py"]
    assert '"Data Manager"' in ui_sources["data_management.py"]
    assert 'DATA_ICON = "storage"' in ui_sources["common.py"]
    assert "DATA_ICON" in ui_sources["overview.py"]
    assert '"database"' not in ui_sources["overview.py"]
    assert "Interface prototype" not in ui_sources["overview.py"]
    for page in ("mapping.py", "simulation.py", "plotting.py"):
        assert "icon=DATA_ICON" in ui_sources[page]
        assert 'panel_title("Controls")' in ui_sources[page]
        assert "stat_strip(" in ui_sources[page]
        assert "preview_plot(" in ui_sources[page]
        assert "raw_output_controls(" in ui_sources[page]
        assert "dn-plot-export-actions" not in ui_sources[page]
        assert 'expansion("Export plot"' not in ui_sources[page]
        for label in (
            "Title",
            "X-axis label",
            "Y-axis label",
            "Line / bar color",
            "Heatmap palette",
            "Histogram bins",
            "Grid",
            "Colorbar",
            "Log Y",
            "Invert Y",
        ):
            assert f'label="{label}"' in ui_sources[page] or f'("{label}"' in ui_sources[page]
    for icon in ("fit_screen", "download", "delete_outline"):
        assert f'icon="{icon}"' in ui_sources["common.py"]
    assert '"Raw data"' in ui_sources["common.py"]
    visible_prototype_copy = (
        "Interface prototype",
        "prototype file",
        "prototype row",
        "interface example",
        "Illustrative preview",
    )
    combined_page_source = "\n".join(ui_sources.values())
    for message in visible_prototype_copy:
        assert message not in combined_page_source
    assert "prototype_badge" not in combined_page_source


def test_styles_include_theme_and_responsive_breakpoints() -> None:
    styles = (APP_DIR / "static" / "styles.css").read_text(encoding="utf-8")
    assert "body.body--dark" in styles
    assert "#b03412" in styles
    assert "@media (max-width: 800px)" in styles
    assert "@media (max-width: 520px)" in styles
    assert "grid-template-columns: minmax(310px, 35fr) minmax(460px, 65fr)" in styles
    assert "scrollbar-gutter: stable" in styles
    assert ".dn-plot-toolbar" in styles
    assert ".dn-plot-tool" in styles
    assert ".dn-upload-compact .q-uploader__header" in styles
    assert ".dn-plot-export-actions" not in styles
    assert ".dn-icon-link > .q-icon" in styles


def test_reused_logo_assets_exist() -> None:
    asset_dir = ROOT_DIR / "docs" / "source" / "_static"
    for name in ("dnascape_logo_small.svg", "dnascape_logo_tab.png"):
        assert (asset_dir / name).is_file(), name
