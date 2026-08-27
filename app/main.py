"""Local entry point and persistent shell for the DNAscape NiceGUI prototype."""

from __future__ import annotations

import os
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.tooltip import Tooltip

from app import __version__
from app.export_settings import (
    CACHE_FILE_FORMAT_OPTIONS,
    COMPRESSION_OPTIONS,
    get_export_settings,
    reset_export_settings,
)
from app.icons import OUTLINE_ICON_SVGS, PLOT_ICON_SVG
from app.pages import PAGE_RENDERERS
from app.session_cache import current_session_cache

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
    # ``None`` is NiceGUI's native automatic mode.  Quasar keeps its media
    # query listener alive, so the interface follows a system-theme change
    # without a reload while still allowing explicit light and dark choices.
    dark_mode = ui.dark_mode(None)
    page_state = {"active": "overview"}
    page_hosts: dict[str, object] = {}
    built_pages: set[str] = set()
    export_settings = get_export_settings()

    def discard_page(slug: str) -> None:
        host = page_hosts.get(slug)
        if host is None:
            return
        cache = current_session_cache()
        cache.unregister_output_clearer(slug)
        cache.unregister_page_refresher(slug)
        host.clear()  # type: ignore[attr-defined]
        built_pages.discard(slug)

    def build_page(slug: str) -> None:
        if slug in built_pages:
            return
        host = page_hosts[slug]
        with host:  # type: ignore[attr-defined]
            PAGE_RENDERERS[slug](navigate)
        built_pages.add(slug)

    def show_page(slug: str) -> None:
        if not export_settings.preserve_page_state:
            discard_page(slug)
        build_page(slug)
        for candidate, host in page_hosts.items():
            host.set_visibility(candidate == slug)  # type: ignore[attr-defined]
        current_session_cache().refresh_page(slug)

    def navigate(slug: str) -> None:
        if slug not in PAGE_RENDERERS or slug == page_state["active"]:
            return
        previous = page_state["active"]
        restore_requested = current_session_cache().has_active(slug)
        if not export_settings.preserve_page_state:
            discard_page(previous)
        if restore_requested:
            discard_page(slug)
        page_state["active"] = slug
        render_navigation.refresh()
        show_page(slug)

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

    def cycle_theme() -> None:
        """Cycle System -> Light -> Dark without losing automatic mode."""

        if dark_mode.value is None:
            dark_mode.disable()
        elif dark_mode.value is False:
            dark_mode.enable()
        else:
            dark_mode.auto()

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
            cached_format_setting = (
                ui.select(
                    CACHE_FILE_FORMAT_OPTIONS,
                    value=export_settings.cached_file_format,
                    label="Cached result file type",
                )
                .props("dense options-dense")
                .classes("dn-field")
                .mark("settings-cached-file-format")
            )
            cached_compression_setting = (
                ui.select(
                    COMPRESSION_OPTIONS,
                    value=export_settings.cached_compression,
                    label="Cached result compression",
                )
                .props("dense options-dense")
                .classes("dn-field")
                .mark("settings-cached-compression")
            )
            preserve_page_setting = (
                ui.checkbox(
                    "Keep page controls and outputs",
                    value=export_settings.preserve_page_state,
                )
                .classes("dn-settings-wide-control")
                .mark("settings-preserve-page-state")
            )

        def update_plot_dpi() -> None:
            value = int(plot_dpi_setting.value or 150)
            export_settings.plot_dpi = max(72, min(384, value))

        def update_compression() -> None:
            export_settings.raw_compression = str(compression_setting.value or "gzip")

        def update_cached_format() -> None:
            export_settings.cached_file_format = str(cached_format_setting.value or "tsv")

        def update_cached_compression() -> None:
            export_settings.cached_compression = str(cached_compression_setting.value or "gzip")

        def update_page_preservation() -> None:
            export_settings.preserve_page_state = bool(preserve_page_setting.value)
            if export_settings.preserve_page_state:
                return
            for slug in tuple(built_pages):
                if slug != page_state["active"]:
                    discard_page(slug)

        plot_dpi_setting.on_value_change(update_plot_dpi)
        compression_setting.on_value_change(update_compression)
        cached_format_setting.on_value_change(update_cached_format)
        cached_compression_setting.on_value_change(update_cached_compression)
        preserve_page_setting.on_value_change(update_page_preservation)

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
                .props('src="/dnascape-assets/dnascape_logo.svg" alt="DNAscape" draggable=false')
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
            ui.link("fp409@cam.ac.uk", "mailto:fp409@cam.ac.uk").classes("dn-about-email").mark(
                "about-email"
            )
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

    def reset_everything() -> None:
        """Return the client-local workspace and shared preferences to defaults."""

        cache = current_session_cache()
        removed = cache.reset()
        for slug in tuple(built_pages):
            discard_page(slug)

        defaults = reset_export_settings()
        plot_dpi_setting.value = defaults.plot_dpi
        compression_setting.value = defaults.raw_compression
        cached_format_setting.value = defaults.cached_file_format
        cached_compression_setting.value = defaults.cached_compression
        preserve_page_setting.value = defaults.preserve_page_state
        for control in (
            plot_dpi_setting,
            compression_setting,
            cached_format_setting,
            cached_compression_setting,
            preserve_page_setting,
        ):
            control.update()

        dark_mode.auto()
        drawer.show()
        sync_drawer_controls()
        page_state["active"] = "overview"
        render_navigation.refresh()
        show_page("overview")
        reset_dialog.close()
        ui.notify(
            f"Defaults restored; cleared {removed} cached item"
            f"{'s' if removed != 1 else ''}.",
            type="positive",
        )

    with (
        ui.dialog().props("persistent") as reset_dialog,
        ui.card().classes("dn-settings-dialog dn-reset-dialog").mark("reset-dialog"),
    ):
        with ui.row().classes("dn-settings-dialog-head"):
            ui.label("Reset DNAscape?").classes("dn-panel-title")
        ui.label(
            "This clears uploaded data, generated results, cached files, and page controls, "
            "then restores application and theme settings."
        ).classes("dn-about-copy")
        with ui.row().classes("dn-dialog-actions"):
            ui.button("Cancel", on_click=reset_dialog.close).props("flat no-caps").mark(
                "cancel-reset-everything"
            )
            ui.button("Reset everything", icon="restart_alt", on_click=reset_everything).props(
                "no-caps"
            ).classes("dn-action-primary").mark("confirm-reset-everything")

    @ui.refreshable
    def render_app_menu_items() -> None:
        theme_label, theme_icon, next_theme = {
            None: ("System theme", "brightness_auto", "light"),
            False: ("Light theme", "light_mode", "dark"),
            True: ("Dark theme", "dark_mode", "system"),
        }[dark_mode.value]
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
            ui.menu_item(on_click=reset_dialog.open)
            .props('aria-label="Reset everything to default"')
            .classes("dn-app-menu-item")
            .mark("menu-reset")
        ):
            with ui.item_section().props("avatar"):
                ui.icon("restart_alt").classes("dn-app-menu-icon")
            with ui.item_section():
                ui.label("Reset to defaults")
        with (
            ui.menu_item(on_click=cycle_theme)
            .props(
                f'aria-label="{theme_label}; switch to {next_theme} theme" '
                f'title="{theme_label}; switch to {next_theme} theme"'
            )
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
        for slug in PAGE_RENDERERS:
            host = ui.element("div").classes("dn-page-view").mark(f"page-host-{slug}")
            host.set_visibility(False)
            page_hosts[slug] = host
        show_page(page_state["active"])


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
