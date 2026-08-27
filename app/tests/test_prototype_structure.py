"""Structural regression tests that do not require importing NiceGUI."""

from __future__ import annotations

import ast
import re
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


def _css_declarations(styles: str, selector: str) -> dict[str, str]:
    """Return declarations for a simple top-level CSS selector."""

    match = re.search(rf"{re.escape(selector)}\s*\{{([^{{}}]*)\}}", styles, re.DOTALL)
    if match is None:
        match = next(
            (
                candidate
                for candidate in re.finditer(r"([^{}]+)\{([^{}]*)\}", styles, re.DOTALL)
                if selector in {item.strip() for item in candidate.group(1).split(",")}
            ),
            None,
        )
    assert match is not None, f"missing CSS selector: {selector}"
    declarations: dict[str, str] = {}
    for declaration in match.group(match.lastindex or 1).split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", 1)
        declarations[name.strip()] = " ".join(value.split())
    return declarations


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


def test_ui_only_imports_scientific_package_through_service_boundaries() -> None:
    scientific_imports: list[tuple[str, str]] = []
    for path in APP_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "dnascape" or alias.name.startswith("dnascape."):
                        scientific_imports.append((path.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = str(node.module)
                if module == "dnascape" or module.startswith("dnascape."):
                    scientific_imports.append((path.name, module))
    assert scientific_imports
    assert {file_name for file_name, _module in scientific_imports} == {
        "data_conversion_service.py",
        "mapping_catalog.py",
        "mapping_service.py",
        "simulation_service.py",
    }


def test_shell_exposes_complete_navigation_and_local_entrypoint() -> None:
    source = (APP_DIR / "main.py").read_text(encoding="utf-8")
    for label in PAGE_LABELS:
        assert f'"{label}"' in source
    assert "127.0.0.1" in source
    assert "dnascape_logo_small.svg" in source
    assert 'mark("nav-logo-overview")' in source
    assert 'mark("navigation-drawer")' in source
    assert 'mark("drawer-toggle")' in source
    assert 'mark("drawer-toggle-slot")' in source
    assert "value=True" in source
    assert "top_corner=True" in source
    assert "top_corner=False" not in source
    assert "drawer.hide()" in source
    assert "drawer.show()" in source
    assert 'mark("drawer-expand")' in source
    assert 'mark("drawer-expand-slot")' in source
    assert 'icon="last_page"' in source
    assert 'drawer.props("mini mini-width=48")' not in source
    assert 'mark("app-menu-toggle")' in source
    assert 'mark("app-menu")' in source
    assert 'mark("menu-theme")' in source
    assert 'mark("theme-action-icon")' in source
    assert 'theme_icon = "light_mode" if use_light_theme else "dark_mode"' in source
    assert 'aria-label="Go to Overview"' in source
    assert "__dnascapeFileDragFeedback" in source
    assert "dn-file-drag-active" in source
    assert "GITHUB_ICON_SVG" in source
    assert 'ui.icon("code")' not in source
    assert '("data-management", "Data Manager", OUTLINE_ICON_SVGS["data"])' in source
    assert '("plotting", "Plotting", None)' in source
    assert "PLOT_ICON_SVG" in source
    assert "dn-nav-outline-icon" in source
    menu_order = tuple(
        source.index(f'"{marker}"')
        for marker in ("menu-theme", "menu-documentation", "menu-github", "menu-issue")
    )
    assert menu_order == tuple(sorted(menu_order))
    assert '"table_view"' not in source
    assert '"database"' not in source
    assert (APP_DIR / "__main__.py").is_file()


def test_workflow_copy_and_data_sections_use_updated_ui_semantics() -> None:
    ui_sources = {
        path.name: path.read_text(encoding="utf-8") for path in (APP_DIR / "pages").glob("*.py")
    }
    preview_source = (APP_DIR / "preview_plotting.py").read_text(encoding="utf-8")
    assert "Data Management" not in "\n".join(ui_sources.values())
    assert '"Data Manager"' in ui_sources["overview.py"]
    assert '"Data Manager"' in ui_sources["data_management.py"]
    icons = (APP_DIR / "icons.py").read_text(encoding="utf-8")
    assert 'DATA_ICON = "dnascape_data"' in icons
    assert 'DATA_ICON_SVG = OUTLINE_ICON_SVGS["data"]' in icons
    assert (
        "from app.icons import DATA_ICON, DATA_ICON_SVG, PLOT_ICON_SVG" in ui_sources["common.py"]
    )
    assert 'section.add_slot("header")' in ui_sources["common.py"]
    assert '"data-outline-icon"' in ui_sources["common.py"]
    assert "OUTLINE_ICON_SVGS" in ui_sources["overview.py"]
    assert 'OUTLINE_ICON_SVGS["data"]' in ui_sources["overview.py"]
    assert "dn-workflow-symbol" in ui_sources["overview.py"]
    assert '"database"' not in ui_sources["overview.py"]
    assert "Interface prototype" not in ui_sources["overview.py"]
    for page in ("data_management.py", "mapping.py", "simulation.py", "plotting.py"):
        assert "icon=DATA_ICON" in ui_sources[page]
        assert 'panel_title("Controls")' in ui_sources[page]
        if page == "data_management.py":
            assert 'panel_title("Download")' in ui_sources[page]
            assert "preview_plot(" not in ui_sources[page]
        else:
            assert "preview_plot(" in ui_sources[page]
        assert "ui.radio(" not in ui_sources[page]
        assert "dn-plot-export-actions" not in ui_sources[page]
        assert 'expansion("Export plot"' not in ui_sources[page]
    for page in ("mapping.py", "simulation.py"):
        assert "action_dock" in ui_sources[page]
        assert "with action_dock():" in ui_sources[page]
        assert "stat_strip(" in ui_sources[page]
        assert "raw_output_controls(" in ui_sources[page]
        preview_title = ui_sources[page].index('panel_title("Preview")')
        plotting_settings = ui_sources[page].index("preview_settings_controls(")
        raw_output = ui_sources[page].index("raw_output_controls(")
        assert preview_title < plotting_settings < raw_output
    plotting_source = ui_sources["plotting.py"]
    assert "with action_dock():" in plotting_source
    assert "stat_strip(" not in plotting_source
    assert "raw_output_controls(" not in plotting_source
    assert 'expansion("Region"' not in plotting_source
    data_section = plotting_source.index('"Data",')
    plotting_settings = plotting_source.index("plot_settings = preview_settings_controls(")
    preview_title = plotting_source.index('panel_title("Preview")')
    assert data_section < plotting_settings < preview_title
    assert "dn-control-plot-settings" in plotting_source
    assert "dn-preview-settings-section" in preview_source
    assert "dn-preview-settings-grid" in preview_source
    for label in (
        "Title",
        "X-axis label",
        "Y-axis label",
        "Curve color",
        "Heatmap palette",
        "Histogram bins",
        "Colorbar",
        "Log Y",
        "Log Z",
        "Invert Y",
    ):
        assert f'label="{label}"' in preview_source or f'("{label}"' in preview_source
    assert 'label="Second curve color"' in preview_source
    assert '"First curve color"' in preview_source
    assert '"Bar color"' in preview_source
    for icon in ("fit_screen", "download"):
        assert f'icon="{icon}"' in ui_sources["common.py"]
    assert 'icon="delete_outline"' not in ui_sources["common.py"]
    assert 'checkbox("Grid"' in preview_source
    assert "def action_dock(" in ui_sources["common.py"]
    assert 'classes("dn-action-dock")' in ui_sources["common.py"]
    assert 'mark("scientific-action-dock")' in ui_sources["common.py"]
    assert '["Profile", "Constant"],' in ui_sources["common.py"]
    assert 'mark(f"source-{marker}-mode")' in ui_sources["common.py"]
    assert "ui.radio(" not in ui_sources["common.py"]
    assert 'f"plotting-profile-source-' in ui_sources["plotting.py"]
    assert "PLOT_ICON_SVG" in ui_sources["overview.py"]
    assert "plot_action_button" in ui_sources["plotting.py"]
    assert "build_preview_plot_options(" in ui_sources["plotting.py"]
    assert "render_preview_chart(" in ui_sources["plotting.py"]
    assert "def data_upload(" in ui_sources["common.py"]
    assert "auto_upload=True" in ui_sources["common.py"]
    assert 'classes("dn-upload-field")' in ui_sources["common.py"]
    assert 'classes("dn-upload-formats")' in ui_sources["common.py"]
    assert '"dragenter"' in ui_sources["common.py"]
    assert '"dragleave"' in ui_sources["common.py"]
    assert "dn-upload-dragging" in ui_sources["common.py"]
    assert 'value="Upload a profile"' in ui_sources["common.py"]
    assert "<polyline points=" not in ui_sources["common.py"]
    assert 'mark("preview-guidance")' in ui_sources["common.py"]
    assert 'marker="data-manager-upload"' in ui_sources["data_management.py"]
    assert "convert_data" in ui_sources["data_management.py"]
    assert "inert_button" not in ui_sources["data_management.py"]
    assert "mini_table(" not in ui_sources["data_management.py"]
    assert '"plotting-data-upload"' in ui_sources["plotting.py"]
    assert sum(source.count("ui.upload(") for source in ui_sources.values()) == 1
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
    assert ".dn-source-compact" in styles
    assert ".dn-source-mode" in styles
    assert ".dn-source-fields" in styles
    assert ".dn-action-dock" in styles
    assert "bottom: 0" in styles
    assert "position: sticky" in styles
    assert ".dn-nav-active" in styles
    assert "font-weight: 400 !important" in styles
    assert ".dn-drawer-toggle" in styles
    assert ".dn-drawer-toggle-slot" in styles
    assert ".dn-drawer-expand-slot" in styles
    assert ".dn-app-menu-button" in styles
    assert ".dn-app-menu-item" in styles
    assert ".dn-main-logo-frame" in styles
    assert "mask-image: radial-gradient" in styles
    assert ".dn-upload-dragging" in styles
    assert ".dn-file-drag-active .dn-upload-compact" in styles
    assert 'content: "check"' not in styles
    assert 'content: "Choose a file"' in styles
    assert 'content: " or drag it here"' in styles
    assert 'content: "cloud_upload"' in styles
    assert ".dn-upload-formats" in styles
    assert ".dn-simulation-settings-grid" in styles
    assert ".dn-setting-group" in styles
    assert ".dn-setting-inline" in styles
    assert ".dn-preview-output-control" in styles


def test_scientific_layout_hooks_keep_controls_aligned_and_actions_clear() -> None:
    styles = (APP_DIR / "static" / "styles.css").read_text(encoding="utf-8")

    control_card = _css_declarations(styles, ".dn-control-card")
    assert control_card["scrollbar-gutter"] == "stable both-edges"
    assert "calc(100dvh - 14rem)" in control_card["max-height"]

    preview_card = _css_declarations(styles, ".dn-preview-card")
    assert preview_card["max-height"] == control_card["max-height"]
    assert preview_card["overflow-y"].startswith("auto")
    assert preview_card["overflow-x"].startswith("hidden")
    assert preview_card["scrollbar-gutter"] == "stable both-edges"
    preview_children = _css_declarations(styles, ".dn-preview-card > *")
    assert preview_children["flex"] == "0 0 auto"
    assert preview_children["min-width"] == "0"

    expansion = _css_declarations(styles, ".dn-expansion")
    assert expansion["width"] == "100%"
    assert expansion["margin"].endswith("!important")
    adjacent_expansion = _css_declarations(styles, ".dn-expansion + .dn-expansion")
    assert adjacent_expansion["margin-top"] != "0"

    # Every expansion uses one quiet header surface and the same full-width
    # alignment contract, regardless of which workspace column owns it.
    expansion_header = _css_declarations(
        styles,
        ".dn-expansion > .q-expansion-item__container > .q-item",
    )
    assert "var(--dn-soft)" in expansion_header["background"]
    assert expansion_header["border-radius"] != "0"

    avatar = _css_declarations(styles, ".dn-expansion .q-item__section--avatar")
    data_icon = _css_declarations(styles, ".dn-expansion-data-icon")
    assert avatar["color"] == "var(--dn-muted)"
    assert data_icon["color"] == "var(--dn-muted)"

    action_dock = _css_declarations(styles, ".dn-action-dock")
    assert action_dock["bottom"] == "0"
    assert action_dock["position"] == "sticky"
    assert "padding" in action_dock

    common = (APP_DIR / "pages" / "common.py").read_text(encoding="utf-8")
    for page_name in ("data_management.py", "mapping.py", "simulation.py", "plotting.py"):
        page = (APP_DIR / "pages" / page_name).read_text(encoding="utf-8")
        assert "with action_dock():" in page
    assert 'classes("dn-action-dock")' in common


def test_navigation_theme_and_compact_height_styles_are_explicit() -> None:
    styles = (APP_DIR / "static" / "styles.css").read_text(encoding="utf-8")

    drawer_toggle = _css_declarations(styles, ".dn-drawer-toggle")
    assert drawer_toggle["background"] == "transparent !important"
    assert drawer_toggle["border"] == "0 !important"
    assert drawer_toggle["box-shadow"] == "none !important"
    assert "@media (hover: hover) and (pointer: fine)" in styles
    assert ".dn-drawer:hover .dn-drawer-toggle" in styles
    assert ".dn-drawer:focus-within .dn-drawer-toggle" in styles
    assert ".dn-drawer-expand-slot" in styles

    drawer_content = _css_declarations(styles, ".dn-drawer-content")
    assert drawer_content["overflow-y"] == "hidden"
    compact_height = styles[styles.index("@media (max-height: 620px)") :]
    compact_drawer_content = _css_declarations(compact_height, ".dn-drawer-content")
    assert compact_drawer_content.get("overflow-y", "hidden") == "hidden"
    compact_nav = _css_declarations(compact_height, ".dn-nav-list")
    assert compact_nav["overflow-y"] == "auto"

    dark_nav = _css_declarations(styles, "body.body--dark .dn-nav-button")
    assert dark_nav["color"] == "var(--dn-text) !important"
    assert "body.body--dark .dn-nav-button .q-btn__content" in styles
    assert "body.body--dark .dn-nav-button .q-btn__content .block" in styles
    assert "body.body--dark .dn-nav-active" in styles

    # Both logo sizes use the same slightly-left, slightly-raised crescent cutout.
    assert re.search(r"circle 2\.5\d*px at 47\.95% 3(?:\.0)?%", styles)
    assert re.search(r"circle 6\.1\d*px at 47\.95% 3(?:\.0)?%", styles)


def test_plotting_and_issue_icons_have_shared_semantics() -> None:
    icons = (APP_DIR / "icons.py").read_text(encoding="utf-8")
    common = (APP_DIR / "pages" / "common.py").read_text(encoding="utf-8")
    main = (APP_DIR / "main.py").read_text(encoding="utf-8")
    overview = (APP_DIR / "pages" / "overview.py").read_text(encoding="utf-8")
    plotting = (APP_DIR / "pages" / "plotting.py").read_text(encoding="utf-8")

    combined = "\n".join((icons, common, main, overview, plotting))
    assert "PLOT_ICON_SVG" in combined
    assert "dn-plot-icon" in combined
    assert "dn-plot-icon-axis" in combined
    assert "dn-plot-icon-line" in combined
    assert 'icon="multiline_chart"' not in plotting
    assert '"plotting", "Plotting", "multiline_chart"' not in main
    assert '"menu-issue"' in main
    assert 'icon="lightbulb"' in main


def test_simulation_output_selector_is_preview_owned_and_settings_are_compact() -> None:
    source = (APP_DIR / "pages" / "simulation.py").read_text(encoding="utf-8")
    export_source = (APP_DIR / "simulation_exports.py").read_text(encoding="utf-8")
    preview_source = (APP_DIR / "preview_plotting.py").read_text(encoding="utf-8")

    assert "Single view" not in source + export_source
    assert 'value="Default"' in preview_source
    assert source.count('.mark("simulation-output")') == 1
    preview_bar = source.index('.mark("simulation-preview-output-bar")')
    output_select = source.index('.mark("simulation-output")')
    assert preview_bar < output_select
    plotting_settings = source.index("preview_settings_controls(")
    raw_data = source.index("raw_output_controls(")
    assert output_select < plotting_settings < raw_data
    assert "sync_plot_labels" in source
    assert "render_latest_result()" in source
    for hook in (
        "dn-simulation-settings-grid",
        "dn-setting-group",
        "dn-setting-group-title",
        "dn-setting-inline",
        "dn-field-compact",
        "dn-field-xs",
        "dn-preview-settings-grid",
        "dn-preview-setting-group",
        "dn-simulation-data-grid",
        "dn-setting-subgroup",
    ):
        assert hook in source or hook in preview_source
    for group in (
        "Firing rate",
        "Fork speed",
        "Ensemble",
        "Timing & outputs",
        "Fork behaviour",
    ):
        assert f'ui.label("{group}")' in source
    for marker in ("simulation-stalling-settings", "simulation-factor-pool-settings"):
        assert f'.mark("{marker}")' in source
    assert 'ui.checkbox("Fixed seed", value=False)' in source
    assert 'ui.checkbox("Y limits"' not in source
    for marker in (
        "simulation-y-minimum",
        "simulation-y-maximum",
        "simulation-reset-y-limits",
    ):
        assert f'"{marker}"' in source
    for marker in (
        "simulation-stats",
        "simulation-finite-pool",
    ):
        assert re.search(rf'\.mark\(\s*["\']{re.escape(marker)}["\']\s*\)', source)


def test_simulation_summary_and_raw_download_css_preserve_alignment() -> None:
    styles = (APP_DIR / "static" / "styles.css").read_text(encoding="utf-8")

    controls = _css_declarations(styles, ".dn-control-card")
    preview = _css_declarations(styles, ".dn-preview-card")
    assert controls["padding"].split()[0] == preview["padding"].split()[0]

    summary = _css_declarations(styles, ".dn-stat-grid.dn-simulation-summary-grid")
    assert summary["grid-template-columns"] == ("minmax(10rem, 1.4fr) repeat(4, minmax(0, 1fr))")
    run_stats = _css_declarations(
        styles,
        ".dn-simulation-summary-grid > .dn-simulation-run-stats",
    )
    assert run_stats["display"] == "contents"

    raw_row = _css_declarations(styles, ".dn-raw-output-row")
    assert raw_row["display"] == "flex"
    assert raw_row["align-items"] == "flex-end"
    download = _css_declarations(styles, ".dn-raw-output-download")
    assert download["margin-left"] == "auto"


def test_reused_logo_assets_exist() -> None:
    asset_dir = ROOT_DIR / "docs" / "source" / "_static"
    for name in ("dnascape_logo_small.svg", "dnascape_logo_tab.png"):
        assert (asset_dir / name).is_file(), name
