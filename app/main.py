"""Local entry point and persistent shell for the DNAscape NiceGUI prototype."""

from __future__ import annotations

import os
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui

from app import __version__
from app.pages import PAGE_RENDERERS
from app.pages.common import DATA_ICON

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
ASSET_DIR = ROOT_DIR / "docs" / "source" / "_static"
STYLE_PATH = APP_DIR / "static" / "styles.css"

NAV_ITEMS = (
    ("overview", "Overview", "home"),
    ("data-management", "Data Manager", DATA_ICON),
    ("mapping", "Mapping", "map"),
    ("simulation", "Simulation", "monitor"),
    ("plotting", "Plotting", "show_chart"),
    ("cached-files", "Cached Files", "inventory_2"),
    ("resources", "Resources", "menu_book"),
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

    # ``top_corner=False`` keeps the global header above the drawer; Quasar's
    # breakpoint still switches it to overlay behavior on narrow screens.
    with ui.left_drawer(value=True, bordered=True, top_corner=False).props(
        "width=240 breakpoint=900"
    ) as drawer:
        drawer.classes("dn-drawer").mark("navigation-drawer")
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
                        button = ui.button(
                            label,
                            icon=icon,
                            on_click=_handler(navigate, slug),
                        ).props("flat no-caps align=left").mark(f"nav-{slug}")
                        button.classes(
                            "dn-nav-button dn-nav-active"
                            if page_state["active"] == slug
                            else "dn-nav-button"
                        )

            render_navigation()
            ui.space()
            with ui.row().classes("dn-sidebar-links"):
                with ui.link(target=GITHUB_URL, new_tab=True).props(
                    'aria-label="GitHub"'
                ).classes("dn-icon-link"):
                    ui.html(GITHUB_ICON_SVG)
                    ui.tooltip("GitHub")
                with ui.link(target=DOCS_URL, new_tab=True).classes("dn-icon-link"):
                    ui.icon("menu_book").tooltip("Documentation")
                with ui.link(target=f"{GITHUB_URL}/issues", new_tab=True).classes("dn-icon-link"):
                    ui.icon("lightbulb").tooltip("Report an issue or suggest a feature")
            ui.label(f"v{__version__.replace('-prototype', '')}").classes("dn-version")

    with ui.header(elevated=False).classes("dn-header"):
        (
            ui.button(icon="menu", on_click=drawer.toggle)
            .props('flat round dense aria-label="Collapse or expand navigation"')
            .classes("dn-menu-button")
            .mark("drawer-toggle")
            .tooltip("Collapse or expand navigation")
        )
        ui.space()
        ui.badge("PROTOTYPE").props("outline").classes("dn-header-badge")
        ui.button(icon="contrast", on_click=dark_mode.toggle).props("flat round dense").classes(
            "dn-theme-button"
        ).tooltip("Toggle light and dark theme")

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
    return os.getenv("DNASCAPE_APP_SHOW", "true").strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
    }


def main() -> None:
    """Start the local-only prototype and open it in the default browser."""

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
