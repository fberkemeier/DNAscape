"""Local entry point and persistent shell for the DNAscape NiceGUI prototype."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import app as nicegui_app
from nicegui import binding, ui
from nicegui.elements.button import Button
from nicegui.elements.tooltip import Tooltip

from app import __version__
from app._startup_cache import read_asset_text
from app.export_settings import (
    DEFAULT_CONTROL_FONT_SIZE_PX,
    get_export_settings,
    reset_export_settings,
)
from app.field_hints import FIELD_HINTS
from app.icons import GITHUB_ICON_SVG, OUTLINE_ICON_SVGS, PLOT_ICON_SVG, RESET_ICON
from app.notifications import notify
from app.page_activity import current_page_activity
from app.pages import PAGE_RENDERERS, preload_workflow_pages
from app.settings_dialog import settings_dialog as create_settings_dialog
from app.theme import brand_logo, logo_sources, theme_css, update_logo

if TYPE_CHECKING:
    from app._launch_progress import LaunchProgress

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
ASSET_DIR = ROOT_DIR / "docs" / "source" / "_static"
STYLE_PATH = APP_DIR / "static" / "styles.css"

NAV_ITEMS = (
    ("overview", "Overview", OUTLINE_ICON_SVGS["overview"]),
    ("data-management", "Data Manager", OUTLINE_ICON_SVGS["data"]),
    ("mapping", "Mapping", OUTLINE_ICON_SVGS["mapping"]),
    ("simulation", "Simulation", OUTLINE_ICON_SVGS["simulation"]),
    ("plotting", "Analysis", None),
    ("cached-files", "Cached Files", OUTLINE_ICON_SVGS["cached"]),
    ("resources", "Resources", OUTLINE_ICON_SVGS["resources"]),
)

GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"

nicegui_app.add_static_files("/dnascape-assets", str(ASSET_DIR))

# Reserve the label line even for blank/automatic values, keeping field text
# and placeholder sizes consistent before focus, during edits and after reset.
for field_type in (ui.input, ui.number, ui.select, ui.textarea):
    field_type.default_props("stack-label clear-icon=close")


def _handler(callback, *args):
    """Return a zero-argument UI handler without capturing event objects."""

    def handle() -> None:
        callback(*args)

    return handle


@ui.page("/")
def index() -> None:
    """Build one isolated, client-local prototype shell."""

    from app.session_cache import current_session_cache

    ui.add_css(read_asset_text(STYLE_PATH))
    ui.add_body_html(
        "<script>" + read_asset_text(STYLE_PATH.parent / "brand_theme.js") + "</script>"
    )
    from dnascape.workflows.plotting import plot_navigation_script
    ui.add_body_html("<script>" + plot_navigation_script() + "</script>")
    ui.add_body_html(
        "<script>window.dnascapeFieldHintLabels=" + json.dumps(FIELD_HINTS) + ";"
        + read_asset_text(STYLE_PATH.parent / "field_hints.js") + "</script>"
    )
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
        <script>
        (() => {
          if (window.__dnascapeFieldHelp) return;
          window.__dnascapeFieldHelp = true;

          const exact = {
            'Prepare as': 'Choose the biological quantity being prepared. This applies sensible value defaults without changing the detected data dependence.',
            'Input type': 'Choose whether this input is supplied as a constant or as one or more profiles.',
            'Value column': 'Choose the data column to use, or All to process every compatible value column.',
            'Input dependence': 'Describe whether values vary over space, time, or both axes.',
            'Output': 'Choose the result currently shown in the preview.',
            'View': 'Choose the visual representation of the selected result.',
            'Layout': 'Choose whether multiple results share one chart or use separate charts.',
            'File name': 'Set the download name without the file extension.',
            'File type': 'Choose the format used when this result is downloaded.',
            'Decimal places': 'Set the number of decimal places written to text-based outputs.',
            'Resolution (kb/bin)': 'Set the width of each spatial bin in kilobases.',
            'Time step (min/bin)': 'Set the interval between adjacent time samples in minutes.',
            'Simulations (replicates)': 'Set the number of independent stochastic simulations.',
            'Seed': 'Set the random seed used when fixed-seed mode is enabled.',
            'Target': 'Choose the quantity that Mapping should infer.',
            'Dependence': 'Choose whether the inferred target varies over space, time, or both.',
            'Route': 'Choose the implemented combination of measured inputs used for the fit.',
            'Bins': 'Set the shared number of intervals used by histogram views.',
            'Bin size': 'Set the shared width of each histogram interval.',
          };

          const descriptionFor = label => {
            if (exact[label]) return exact[label];
            const key = label.toLowerCase();
            if (key.includes('resolution')) return 'Set the sampling resolution used for this axis or operation.';
            if (key.includes('minimum') || key.endsWith(' min')) return `Set the lower bound for ${label.toLowerCase()}.`;
            if (key.includes('maximum') || key.endsWith(' max')) return `Set the upper bound for ${label.toLowerCase()}.`;
            if (key.includes('column')) return 'Choose which compatible column from the loaded table is used.';
            if (key.includes('color') || key.includes('colour')) return 'Choose the colour used for this curve or plot element.';
            if (key.includes('width') || key.includes('size')) return `Adjust ${label.toLowerCase()} for the selected output.`;
            if (key.includes('start') || key.includes('end')) return `Set the ${label.toLowerCase()} of the selected range.`;
            if (key.includes('rate')) return `Set ${label.toLowerCase()} in the units shown by the field.`;
            if (key.includes('time')) return `Set ${label.toLowerCase()} in the units shown by the field.`;
            return `Set ${label.toLowerCase()} for this workflow.`;
          };

          const decorate = () => {
            document.querySelectorAll('.dn-field').forEach(field => {
              const labelNode = field.querySelector('.q-field__label');
              const label = (field.dataset.helpLabel || labelNode?.textContent || '').trim();
              if (!label) return;
              let icon = Array.from(field.children).find(
                child => child.classList?.contains('dn-field-help-icon'),
              );
              if (!icon) {
                icon = document.createElement('span');
                icon.className = 'dn-field-help-icon';
                icon.textContent = '?';
                icon.tabIndex = 0;
                field.appendChild(icon);
              }
              const description = field.dataset.help || descriptionFor(label);
              icon.title = description;
              icon.setAttribute('aria-label', `Help for ${label}: ${description}`);
            });
          };

          let scheduled = false;
          const schedule = () => {
            if (scheduled) return;
            scheduled = true;
            window.requestAnimationFrame(() => {
              scheduled = false;
              decorate();
            });
          };
          new MutationObserver(schedule).observe(document.documentElement, {
            childList: true,
            subtree: true,
            characterData: true,
          });
          if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', schedule, {once: true});
          } else {
            schedule();
          }
        })();
        </script>
        """
    )
    ui.colors(primary="var(--dn-accent)")
    body = ui.query("body").classes("dn-body")
    # ``None`` is NiceGUI's native automatic mode.  Quasar keeps its media
    # query listener alive, so the interface follows a system-theme change
    # without a reload while still allowing explicit light and dark choices.
    dark_mode = ui.dark_mode(None)
    page_state = {"active": "overview"}
    page_hosts: dict[str, object] = {}
    built_pages: set[str] = set()
    page_activity = current_page_activity()
    page_activity.is_page_active = lambda slug: page_state["active"] == slug
    export_settings = get_export_settings()

    def sync_control_font_size() -> None:
        size = export_settings.control_font_size_px
        body.style(
            f"--dn-control-font-size: {size}px; "
            f"--dn-control-font-scale: {size / DEFAULT_CONTROL_FONT_SIZE_PX:.6g};"
        )

    def sync_theme() -> None:
        color = export_settings.theme_color
        body.style("; ".join(f"{key}: {value}" for key, value in theme_css(color).items()))
        for element in tuple(ui.context.client.elements.values()):
            if asset := element._props.get("data-dn-logo"):
                update_logo(element, asset, color)
        light, dark = logo_sources("dnascape_logo_tab.svg", color)
        body.props(f'data-dn-favicon-light="{light}" data-dn-favicon-dark="{dark}"')

    sync_control_font_size()
    sync_theme()
    if export_settings.show_help:
        body.classes("dn-field-help-enabled")
    if not export_settings.show_field_hints:
        body.classes("dn-field-hints-disabled")

    def discard_page(slug: str) -> None:
        if page_activity.retain_page(slug):
            return
        host = page_hosts.get(slug)
        if host is None:
            return
        cache = current_session_cache()
        cache.unregister_output_clearer(slug)
        cache.unregister_page_refresher(slug)
        cache.unregister_input_targets(slug)
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
        if slug not in PAGE_RENDERERS:
            return
        if slug == page_state["active"]:
            page_activity.acknowledge(slug)
            return
        previous = page_state["active"]
        restore_requested = current_session_cache().has_active(slug)
        if not export_settings.preserve_page_state:
            discard_page(previous)
        if restore_requested:
            # An explicit cached-result choice replaces unread output, while
            # the retention guard still protects a job which is running.
            page_activity.acknowledge(slug)
            discard_page(slug)
        page_state["active"] = slug
        render_navigation.refresh()
        show_page(slug)
        page_activity.acknowledge(slug)

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
                brand_logo(
                    "dnascape_logo_small.svg", alt="DNAscape — return to Overview",
                ).classes("dn-sidebar-logo")

            @ui.refreshable
            def render_navigation() -> None:
                def render_activity(slug: str, label: str) -> None:
                    indicator = page_activity.indicator(slug)
                    if indicator is None:
                        return
                    with ui.element("span").classes("dn-nav-status").props("role=status"):
                        if indicator == "running":
                            ui.spinner(size="14px", color="primary").props(
                                f'aria-label="{label} running"'
                            ).mark(f"nav-{slug}-running")
                        else:
                            ui.icon("check", size="1em", color="primary").props(
                                f'aria-label="{label} complete"'
                            ).mark(f"nav-{slug}-complete")

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
                                render_activity(slug, label)
                        else:
                            assert icon is not None
                            with (
                                ui.button(on_click=_handler(navigate, slug))
                                .props("flat no-caps align=left")
                                .mark(f"nav-{slug}")
                            ) as button:
                                ui.html(icon).classes("dn-nav-symbol dn-nav-outline-icon")
                                ui.label(label).classes("dn-nav-label")
                                render_activity(slug, label)
                        button.classes(
                            "dn-nav-button dn-nav-active"
                            if page_state["active"] == slug
                            else "dn-nav-button"
                        )

            page_activity.on_change = render_navigation.refresh
            render_navigation()
            ui.space()
            with ui.row().classes("dn-sidebar-links"):
                with (
                    ui.link(target=GITHUB_URL, new_tab=True)
                    .props('aria-label="GitHub"')
                    .classes("dn-icon-link")
                ):
                    ui.html(GITHUB_ICON_SVG)
                    ui.tooltip("GitHub").props('anchor="top middle" self="bottom middle"')
                with ui.link(target=DOCS_URL, new_tab=True).classes("dn-icon-link"):
                    ui.icon("menu_book")
                    ui.tooltip("Documentation").props('anchor="top middle" self="bottom middle"')
                with ui.link(target=f"{GITHUB_URL}/issues", new_tab=True).classes("dn-icon-link"):
                    ui.icon("lightbulb")
                    ui.tooltip("Report an issue or suggest a feature").props(
                        'anchor="top middle" self="bottom middle"'
                    )
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

    applied_preview_points = export_settings.preview_points

    def apply_application_settings() -> None:
        nonlocal applied_preview_points
        sync_control_font_size()
        sync_theme()
        body.classes(
            add="dn-field-help-enabled" if export_settings.show_help else None,
            remove=None if export_settings.show_help else "dn-field-help-enabled",
        )
        body.classes(
            add=None if export_settings.show_field_hints else "dn-field-hints-disabled",
            remove="dn-field-hints-disabled" if export_settings.show_field_hints else None,
        )
        for slug in tuple(built_pages):
            if not export_settings.preserve_page_state and slug != page_state["active"]:
                discard_page(slug)
            else:
                current_session_cache().refresh_page(slug)
        if applied_preview_points != export_settings.preview_points:
            from app.preview_plotting import refresh_preview_points
            refresh_preview_points()
            applied_preview_points = export_settings.preview_points

    settings_dialog = create_settings_dialog(on_apply=apply_application_settings)

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
        with ui.element("div").classes("dn-about-content"):
            with ui.element("div").classes("dn-about-identity"):
                (
                    brand_logo("dnascape_logo.svg")
                    .classes("dn-about-logo")
                    .mark("about-logo")
                )
                ui.label(f"Version {__version__.replace('-prototype', '')}").classes(
                    "dn-about-version"
                )
            ui.label(
                "DNAscape is open-source software for quantitative modelling of DNA replication. "
                "It brings genomic data preparation, kinetic inference, replication simulations "
                "and statistical analysis into one workspace, helping you compare models with "
                "experiments and test mechanistic hypotheses."
            ).classes("dn-about-copy dn-about-summary")
            with ui.element("div").classes("dn-about-footer"):
                with ui.element("div").classes("dn-about-author"):
                    ui.label("Created and maintained by").classes("dn-about-maintenance")
                    ui.label("Francisco Berkemeier").classes("dn-about-author-name")
                    ui.link("fp409@cam.ac.uk", "mailto:fp409@cam.ac.uk").classes("dn-about-email").mark(
                        "about-email"
                    )
                with ui.element("div").classes("dn-about-links"):
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
            with ui.element("div").classes("dn-about-legal"):
                ui.label("© 2026 Francisco Berkemeier").classes("dn-about-copyright")
                (
                    ui.link("Released under the MIT licence", f"{GITHUB_URL}/blob/main/LICENSE", new_tab=True)
                    .classes("dn-about-licence")
                    .mark("about-licence")
                )

    def reset_everything() -> None:
        """Return the client-local workspace and shared preferences to defaults."""

        cache = current_session_cache()
        page_activity.reset()
        removed = cache.reset()
        for slug in tuple(built_pages):
            discard_page(slug)

        defaults = reset_export_settings()
        sync_control_font_size()
        sync_theme()
        settings_dialog.load(defaults)
        body.classes(
            add="dn-field-help-enabled" if defaults.show_help else None,
            remove=None if defaults.show_help else "dn-field-help-enabled",
        )
        body.classes(
            add=None if defaults.show_field_hints else "dn-field-hints-disabled",
            remove="dn-field-hints-disabled" if defaults.show_field_hints else None,
        )

        dark_mode.auto()
        drawer.show()
        sync_drawer_controls()
        page_state["active"] = "overview"
        render_navigation.refresh()
        show_page("overview")
        reset_dialog.close()
        notify(
            f"Defaults restored; cleared {removed} cached item"
            f"{'s' if removed != 1 else ''}.",
            type="positive",
        )

    with (
        ui.dialog().mark("reset-confirmation-dialog") as reset_dialog,
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
            ui.button("Reset everything", icon=RESET_ICON, on_click=reset_everything).props(
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
            ui.menu_item(on_click=reset_dialog.open)
            .props('aria-label="Reset everything to default"')
            .classes("dn-app-menu-item")
            .mark("menu-reset")
        ):
            with ui.item_section().props("avatar"):
                ui.icon(RESET_ICON).classes("dn-app-menu-icon")
            with ui.item_section():
                ui.label("Reset to defaults")
        with (
            ui.menu_item(on_click=cycle_theme, auto_close=False)
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
            with (
                ui.menu()
                .props('anchor="bottom right" self="top right"')
                .classes("dn-app-menu")
                .mark("app-menu")
            ) as app_menu:
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


def main(*, launch_progress: LaunchProgress | None = None) -> None:
    """Start the local-only interface; browser launch is explicitly opt-in."""

    # Plot refreshes can exceed NiceGUI's 10 ms diagnostic threshold briefly.
    binding.MAX_PROPAGATION_TIME = 0.1
    if launch_progress is not None:
        launch_progress.update(1, "Loading workflows")
    preload_workflow_pages()
    if launch_progress is not None:
        launch_progress.update(2, "Starting server")

        async def launch_ready() -> None:
            from nicegui.server import Server

            # ASGI startup hooks precede Uvicorn's socket binding. Keep the
            # indicator active until this server has actually started.
            while not Server.instance.started:
                await asyncio.sleep(.05)
            launch_progress.ready(nicegui_app.urls)

        nicegui_app.on_startup(launch_ready)
    ui.run(
        title="DNAscape",
        favicon=str(ASSET_DIR / "dnascape_logo_tab.png"),
        host=os.getenv("DNASCAPE_APP_HOST", "127.0.0.1"),
        port=_port(),
        reload=False,
        show=_show_browser(),
        show_welcome_message=launch_progress is None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
