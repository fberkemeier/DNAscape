"""Local entry point and persistent shell for the DNAscape NiceGUI prototype."""

from __future__ import annotations

import os
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.tooltip import Tooltip

from app import __version__
from app.export_settings import COMPRESSION_OPTIONS, get_export_settings
from app.icons import OUTLINE_ICON_SVGS, PLOT_ICON_SVG
from app.pages import PAGE_RENDERERS

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
ASSET_DIR = ROOT_DIR / "docs" / "source" / "_static"
STYLE_PATH = APP_DIR / "static" / "styles.css"

NAV_ITEMS = (
    ("overview", "Overview", OUTLINE_ICON_SVGS["overview"]),
    ("data-management", "Data Manager", OUTLINE_ICON_SVGS["data"]),
    ("mapping", "Mapping", OUTLINE_ICON_SVGS["mapping"]),
    ("simulation", "Simulation", OUTLINE_ICON_SVGS["simulation"]),
    ("plotting", "Plotting", None),
    ("cached-files", "Cached Files", OUTLINE_ICON_SVGS["cached"]),
    ("resources", "Resources", OUTLINE_ICON_SVGS["resources"]),
)

GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"

GITHUB_ICON_SVG = """
<svg class="dn-github-icon" viewBox="0 0 16 16" width="18" height="18"
     aria-hidden="true" focusable="false">
  <path fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
           0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
           -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
           2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
           0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
           2.2.82A7.61 7.61 0 0 1 8 3.86c.68 0 1.36.09 2 .27
           1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
           0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
           0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8
           c0-4.42-3.58-8-8-8Z"/>
</svg>
"""

nicegui_app.add_static_files("/dnascape-assets", str(ASSET_DIR))


def _handler(callback, *args):
    """Return a zero-argument UI handler without capturing event objects."""

    def handle() -> None:
        callback(*args)

    return handle


@ui.page("/")
def index() -> None:
    """Build one isolated, client-local prototype shell."""

    ui.add_css(STYLE_PATH.read_text(encoding="utf-8"))
    ui.add_body_html(
        """
        <script>
        (() => {
          if (window.__dnascapeFileDragFeedback) return;
          window.__dnascapeFileDragFeedback = true;
          let clearTimer = null;
          const hasFiles = event =>
            Array.from(event.dataTransfer?.types || []).includes('Files');
          const show = event => {
            if (!hasFiles(event) || !document.querySelector('.dn-upload-compact')) return;
            event.preventDefault();
            if (clearTimer !== null) window.clearTimeout(clearTimer);
            clearTimer = null;
            document.body.classList.add('dn-file-drag-active');
          };
          const clear = () => {
            if (clearTimer !== null) window.clearTimeout(clearTimer);
            clearTimer = null;
            document.body.classList.remove('dn-file-drag-active');
          };
          document.addEventListener('dragenter', show, true);
          document.addEventListener('dragover', show, true);
          document.addEventListener('dragleave', () => {
            if (clearTimer !== null) window.clearTimeout(clearTimer);
            clearTimer = window.setTimeout(clear, 90);
          }, true);
          document.addEventListener('drop', event => {
            if (hasFiles(event)) event.preventDefault();
            clear();
          }, true);
          document.addEventListener('dragend', clear, true);
          window.addEventListener('blur', clear);
        })();
        </script>
        """
    )
    ui.colors(primary="#b03412")
    ui.query("body").classes("dn-body")
    dark_mode = ui.dark_mode()
    page_state = {"active": "overview"}

    def navigate(slug: str) -> None:
        if slug not in PAGE_RENDERERS or slug == page_state["active"]:
            return
        page_state["active"] = slug
        render_navigation.refresh()
        render_active_page.refresh()

    drawer_collapse_button: Button
    drawer_collapse_tooltip: Tooltip
    drawer_expand_button: Button
    drawer_expand_tooltip: Tooltip

    def sync_drawer_controls() -> None:
        """Keep the two spatially distinct drawer controls in sync."""

        drawer_is_open = drawer.value is not False
        drawer_collapse_button.set_visibility(drawer_is_open)
        drawer_expand_button.set_visibility(not drawer_is_open)
        drawer_collapse_button.props["aria-expanded"] = str(drawer_is_open).lower()
        drawer_expand_button.props["aria-expanded"] = str(drawer_is_open).lower()
        drawer_collapse_tooltip.set_text("Collapse navigation")
        drawer_expand_tooltip.set_text("Expand navigation")

    def collapse_drawer() -> None:
        drawer.hide()
        sync_drawer_controls()

    def expand_drawer() -> None:
        drawer.show()
        sync_drawer_controls()

    # The drawer occupies the full left edge, including the header corner.
    # Quasar's breakpoint still switches it to overlay behavior on narrow screens.
    with ui.left_drawer(value=True, bordered=True, top_corner=True).props(
        'width=240 breakpoint=900 id="dn-navigation-drawer"'
    ) as drawer:
        drawer.classes("dn-drawer").mark("navigation-drawer")
        with ui.element("div").classes("dn-drawer-toggle-slot").mark("drawer-toggle-slot"):
            with (
                ui.button(icon="first_page", on_click=collapse_drawer)
                .props(
                    'flat round dense aria-label="Collapse navigation" '
                    'aria-controls="dn-navigation-drawer" aria-expanded="true"'
                )
                .classes("dn-drawer-toggle dn-drawer-collapse")
                .mark("drawer-toggle")
            ) as drawer_collapse_button:
                drawer_collapse_tooltip = ui.tooltip("Collapse navigation")
        with ui.column().classes("dn-drawer-content"):
            with (
                ui.element("button")
                .props('type=button aria-label="Go to Overview"')
                .classes("dn-logo-link")
                .on("click", _handler(navigate, "overview"))
                .mark("nav-logo-overview")
            ):
                ui.element("img").props(
                    'src="/dnascape-assets/dnascape_logo_small.svg" '
                    'alt="DNAscape — return to Overview" draggable=false'
                ).classes("dn-sidebar-logo")
                ui.tooltip("Go to Overview")

            @ui.refreshable
            def render_navigation() -> None:
                with ui.column().classes("dn-nav-list"):
                    for slug, label, icon in NAV_ITEMS:
                        if slug == "plotting":
                            with (
                                ui.button(on_click=_handler(navigate, slug))
                                .props("flat no-caps align=left")
                                .mark(f"nav-{slug}")
                            ) as button:
                                ui.html(PLOT_ICON_SVG).classes("dn-nav-plot-icon dn-plot-icon-wrap")
                                ui.label(label).classes("dn-nav-label")
                        else:
                            assert icon is not None
                            with (
                                ui.button(on_click=_handler(navigate, slug))
                                .props("flat no-caps align=left")
                                .mark(f"nav-{slug}")
                            ) as button:
                                ui.html(icon).classes("dn-nav-symbol dn-nav-outline-icon")
                                ui.label(label).classes("dn-nav-label")
                        button.classes(
                            "dn-nav-button dn-nav-active"
                            if page_state["active"] == slug
                            else "dn-nav-button"
                        )

            render_navigation()
            ui.space()
            with ui.row().classes("dn-sidebar-links"):
                with (
                    ui.link(target=GITHUB_URL, new_tab=True)
                    .props('aria-label="GitHub"')
                    .classes("dn-icon-link")
                ):
                    ui.html(GITHUB_ICON_SVG)
                    ui.tooltip("GitHub")
                with ui.link(target=DOCS_URL, new_tab=True).classes("dn-icon-link"):
                    ui.icon("menu_book").tooltip("Documentation")
                with ui.link(target=f"{GITHUB_URL}/issues", new_tab=True).classes("dn-icon-link"):
                    ui.icon("lightbulb").tooltip("Report an issue or suggest a feature")
            ui.label(f"v{__version__.replace('-prototype', '')}").classes("dn-version")

    # Once the drawer has fully closed, its own control is no longer reachable.
    # This small fixed control deliberately lives outside the drawer so opening
    # navigation never depends on a remnant rail or an invisible hover target.
    with ui.element("div").classes("dn-drawer-expand-slot").mark("drawer-expand-slot"):
        with (
            ui.button(icon="last_page", on_click=expand_drawer)
            .props(
                'flat round dense aria-label="Expand navigation" '
                'aria-controls="dn-navigation-drawer" aria-expanded="false"'
            )
            .classes("dn-drawer-toggle dn-drawer-expand")
            .mark("drawer-expand")
        ) as drawer_expand_button:
            drawer_expand_tooltip = ui.tooltip("Expand navigation")

    drawer.on_value_change(lambda: sync_drawer_controls())
    sync_drawer_controls()

    def toggle_theme() -> None:
        dark_mode.toggle()

    def open_external(target: str) -> None:
        ui.navigate.to(target, new_tab=True)

    def external_menu_item(
        label: str,
        marker: str,
        target: str,
        *,
        icon: str | None = None,
        github_icon: bool = False,
    ) -> None:
        with (
            ui.menu_item(on_click=_handler(open_external, target))
            .props(f'aria-label="{label} (opens in a new tab)"')
            .classes("dn-app-menu-item")
            .mark(marker)
        ):
            with ui.item_section().props("avatar"):
                if github_icon:
                    ui.html(GITHUB_ICON_SVG).classes("dn-app-menu-icon")
                elif icon is not None:
                    ui.icon(icon).classes("dn-app-menu-icon")
            with ui.item_section():
                ui.label(label)

    export_settings = get_export_settings()
    with (
        ui.dialog().props("persistent") as settings_dialog,
        ui.card().classes("dn-settings-dialog"),
    ):
        with ui.row().classes("dn-settings-dialog-head"):
            ui.label("Settings").classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=settings_dialog.close).props(
                'flat round dense aria-label="Close settings"'
            )
        with ui.element("div").classes("dn-settings-grid"):
            plot_dpi_setting = (
                ui.number(
                    label="Plot export DPI",
                    value=export_settings.plot_dpi,
                    min=72,
                    max=384,
                    step=24,
                )
                .props("dense")
                .classes("dn-field")
                .mark("settings-plot-dpi")
            )
            compression_setting = (
                ui.select(
                    COMPRESSION_OPTIONS,
                    value=export_settings.raw_compression,
                    label="Raw data compression",
                )
                .props("dense options-dense")
                .classes("dn-field")
                .mark("settings-raw-compression")
            )

        def update_plot_dpi() -> None:
            value = int(plot_dpi_setting.value or 150)
            export_settings.plot_dpi = max(72, min(384, value))

        def update_compression() -> None:
            export_settings.raw_compression = str(compression_setting.value or "gzip")

        plot_dpi_setting.on_value_change(update_plot_dpi)
        compression_setting.on_value_change(update_compression)

    with (
        ui.dialog() as about_dialog,
        ui.card().classes("dn-settings-dialog dn-about-dialog").mark("about-dialog"),
    ):
        with ui.row().classes("dn-settings-dialog-head"):
            ui.label("About").classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=about_dialog.close).props(
                'flat round dense aria-label="Close About"'
            )
        with ui.element("div").classes("dn-about-identity"):
            (
                ui.element("img")
                .props(
                    'src="/dnascape-assets/dnascape_logo.svg" '
                    'alt="DNAscape" draggable=false'
                )
                .classes("dn-about-logo")
                .mark("about-logo")
            )
            with ui.row().classes("dn-about-meta"):
                ui.label(f"Version {__version__.replace('-prototype', '')}").classes(
                    "dn-about-version"
                )
                ui.label("Local Python interface").classes("dn-about-local")
        ui.label(
            "DNAscape builds quantitative models of DNA replication from genomic data. "
            "It connects data preparation, kinetic mapping, stochastic simulation, and "
            "scientific visualization in one local workspace."
        ).classes("dn-about-copy dn-about-summary")
        with ui.element("div").classes("dn-about-capabilities"):
            for icon, label in (
                ("swap_horiz", "Map measured profiles to replication kinetics"),
                ("science", "Simulate replication programmes with rsim"),
                ("query_stats", "Inspect spatial and temporal outputs"),
            ):
                with ui.row().classes("dn-about-capability"):
                    ui.icon(icon)
                    ui.label(label)
        ui.separator().classes("dn-about-separator")
        with ui.element("div").classes("dn-about-author"):
            ui.label("Created and maintained by Francisco Berkemeier").classes(
                "dn-about-author-name"
            )
            ui.link("fp409@cam.ac.uk", "mailto:fp409@cam.ac.uk").classes(
                "dn-about-email"
            ).mark("about-email")
        with ui.row().classes("dn-about-links"):
            with (
                ui.link(target=DOCS_URL, new_tab=True)
                .classes("dn-about-link")
                .mark("about-documentation")
            ):
                ui.icon("menu_book")
                ui.label("Documentation")
            with (
                ui.link(target=GITHUB_URL, new_tab=True)
                .classes("dn-about-link")
                .mark("about-github")
            ):
                ui.html(GITHUB_ICON_SVG).classes("dn-about-github-icon")
                ui.label("GitHub")
        ui.label("Copyright 2026 Francisco Berkemeier. All rights reserved.").classes(
            "dn-about-copyright"
        )

    @ui.refreshable
    def render_app_menu_items() -> None:
        use_light_theme = bool(dark_mode.value)
        theme_label = "Light theme" if use_light_theme else "Dark theme"
        theme_icon = "light_mode" if use_light_theme else "dark_mode"
        with (
            ui.menu_item(on_click=settings_dialog.open)
            .props('aria-label="Open settings"')
            .classes("dn-app-menu-item")
            .mark("menu-settings")
        ):
            with ui.item_section().props("avatar"):
                ui.icon("settings").classes("dn-app-menu-icon")
            with ui.item_section():
                ui.label("Settings")
        with (
            ui.menu_item(on_click=about_dialog.open)
            .props('aria-label="Open About DNAscape"')
            .classes("dn-app-menu-item")
            .mark("menu-about")
        ):
            with ui.item_section().props("avatar"):
                ui.icon("info").classes("dn-app-menu-icon")
            with ui.item_section():
                ui.label("About")
        with (
            ui.menu_item(on_click=toggle_theme)
            .props(f'aria-label="Switch to {theme_label.lower()}"')
            .classes("dn-app-menu-item")
            .mark("menu-theme")
        ):
            with ui.item_section().props("avatar"):
                ui.icon(theme_icon).classes("dn-app-menu-icon").mark("theme-action-icon")
            with ui.item_section():
                ui.label(theme_label)
        external_menu_item(
            "Documentation",
            "menu-documentation",
            DOCS_URL,
            icon="menu_book",
        )
        external_menu_item("GitHub", "menu-github", GITHUB_URL, github_icon=True)
        external_menu_item(
            "Raise an issue",
            "menu-issue",
            f"{GITHUB_URL}/issues",
            icon="lightbulb",
        )

    dark_mode.on_value_change(lambda: render_app_menu_items.refresh())

    with ui.header(elevated=False).classes("dn-header"):
        ui.space()
        ui.badge("PROTOTYPE").props("outline").classes("dn-header-badge")
        with (
            ui.button(icon="menu")
            .props('flat round dense aria-label="Open application menu"')
            .classes("dn-app-menu-button")
            .mark("app-menu-toggle")
        ):
            ui.tooltip("Application menu")
            with (
                ui.menu()
                .props('anchor="bottom right" self="top right"')
                .classes("dn-app-menu")
                .mark("app-menu")
            ):
                render_app_menu_items()

    with ui.column().classes("dn-page-host"):

        @ui.refreshable
        def render_active_page() -> None:
            renderer = PAGE_RENDERERS.get(page_state["active"], PAGE_RENDERERS["overview"])
            renderer(navigate)

        render_active_page()


def _port() -> int:
    try:
        port = int(os.getenv("DNASCAPE_APP_PORT", "8080"))
    except ValueError:
        return 8080
    return port if 1 <= port <= 65_535 else 8080


def _show_browser() -> bool:
    return os.getenv("DNASCAPE_APP_SHOW", "false").strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
    }


def main() -> None:
    """Start the local-only interface; browser launch is explicitly opt-in."""

    ui.run(
        title="DNAscape",
        favicon=str(ASSET_DIR / "dnascape_logo_tab.png"),
        host=os.getenv("DNASCAPE_APP_HOST", "127.0.0.1"),
        port=_port(),
        reload=False,
        show=_show_browser(),
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
