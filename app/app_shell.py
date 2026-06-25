"""Shared layout helpers for the DNAscape Streamlit prototype."""

import base64
import html
import importlib.metadata
import json
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for import_path in (ROOT_DIR, APP_DIR):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)
LOGO_PATH = ROOT_DIR / "docs" / "source" / "_static" / "dnascape_logo.png"
SIDEBAR_LOGO_PATH = ROOT_DIR / "docs" / "source" / "_static" / "dnascape_logo_small.png"
TAB_LOGO_PATH = ROOT_DIR / "docs" / "source" / "_static" / "dnascape_logo_tab.png"
GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"
PACKAGE_NAME = "dnascape"
ACTIVE_PAGE_KEY = "dnascape_active_page"
LAST_QUERY_PAGE_KEY = "dnascape_last_query_page"
NAV_ITEMS = (
    ("overview", "Overview", "/"),
    ("data-management", "Data Management", "/Data_Management"),
    ("mapping", "Mapping", "/Mapping"),
    ("simulation", "Simulation", "/Simulation"),
    ("plotting", "Plotting", "/Plotting"),
    ("cached-files", "Cached Files", "/Cached_Files"),
    ("resources", "Resources", "/Resources"),
)
PAGE_LABELS = {slug: label for slug, label, _ in NAV_ITEMS}
PAGE_SLUGS = {
    label.casefold(): slug
    for slug, label, _ in NAV_ITEMS
}
PAGE_SLUGS.update({slug: slug for slug, _, _ in NAV_ITEMS})
THEMES = ("light", "dark", "system")
THEME_ICONS = {
    "light": """
      <svg class="dnascape-theme-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <circle cx="12" cy="12" r="4.2" fill="none" stroke="currentColor" stroke-width="1.8"/>
        <path d="M12 2.8v2.2M12 19v2.2M4.2 4.2l1.55 1.55M18.25 18.25l1.55 1.55M2.8 12h2.2M19 12h2.2M4.2 19.8l1.55-1.55M18.25 5.75l1.55-1.55" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
      </svg>
    """,
    "dark": """
      <svg class="dnascape-theme-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M20.15 14.35A7.7 7.7 0 0 1 9.65 3.85a8 8 0 1 0 10.5 10.5Z" fill="currentColor"/>
      </svg>
    """,
    "system": """
      <svg class="dnascape-theme-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M12 3.8a8.2 8.2 0 0 1 0 16.4Z" fill="currentColor"/>
        <circle cx="12" cy="12" r="8.2" fill="none" stroke="currentColor" stroke-width="1.8"/>
        <path d="M12 3.8v16.4" fill="none" stroke="currentColor" stroke-width="1.8"/>
      </svg>
    """,
}
THEME_LABELS = {
    "light": "Light",
    "dark": "Dark",
    "system": "System",
}


def package_version() -> str:
    """Read the installed package version, falling back to pyproject.toml."""
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        pyproject_path = ROOT_DIR / "pyproject.toml"
        if pyproject_path.exists():
            for line in pyproject_path.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "version":
                    return value.strip().strip('"').strip("'")
        return "0.0.0"


APP_VERSION = package_version()


def normalize_page(value: object, default: str = "overview") -> str:
    """Return a known router slug from a label, slug, or URL-ish value."""
    if isinstance(value, list):
        value = value[0] if value else ""

    raw = str(value or "").strip()
    if not raw:
        return default

    compact = raw.split("?", 1)[0].strip("/").replace("_", "-").replace(" ", "-")
    key = compact.casefold()
    return PAGE_SLUGS.get(key, default)


def active_page_slug(default: str = "overview") -> str:
    """Resolve the active page from session state, with query params as bootstrap."""
    query_page = normalize_page(st.query_params.get("page"), default="")
    if query_page and st.session_state.get(LAST_QUERY_PAGE_KEY) != query_page:
        st.session_state[ACTIVE_PAGE_KEY] = query_page
        st.session_state[LAST_QUERY_PAGE_KEY] = query_page

    if ACTIVE_PAGE_KEY not in st.session_state:
        st.session_state[ACTIVE_PAGE_KEY] = default

    return normalize_page(st.session_state[ACTIVE_PAGE_KEY], default=default)


def active_page_label(default: str = "overview") -> str:
    """Return the display label for the active router page."""
    return PAGE_LABELS[active_page_slug(default)]


def set_active_page(slug: str) -> None:
    """Select a page in the single-script Streamlit router."""
    st.session_state[ACTIVE_PAGE_KEY] = normalize_page(slug)


def configure_page(title: str) -> None:
    """Apply Streamlit page settings and shared visual styling."""
    st.set_page_config(
        page_title=title,
        page_icon=TAB_LOGO_PATH,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": DOCS_URL,
            "Report a bug": f"{GITHUB_URL}/issues",
            "About": "DNAscape prototype workspace.",
        },
    )
    apply_style()
    render_theme_toggle()


def apply_style() -> None:
    """Keep the prototype calm, sparse, and workspace-like."""
    st.markdown(
        """
        <style>
        :root {
          --dnascape-bg: var(--background-color, #f7f7f8);
          --dnascape-panel: var(--secondary-background-color, #ffffff);
          --dnascape-text: var(--text-color, #403932);
          --dnascape-accent: var(--primary-color, #b03412);
          --dnascape-muted: color-mix(in srgb, var(--dnascape-text) 72%, var(--dnascape-bg));
          --dnascape-line: color-mix(in srgb, var(--dnascape-text) 14%, var(--dnascape-bg));
          --dnascape-divider: color-mix(in srgb, var(--dnascape-text) 24%, var(--dnascape-bg));
          --dnascape-sidebar-bg: color-mix(in srgb, var(--dnascape-bg) 92%, var(--dnascape-panel));
          --dnascape-hover-bg: color-mix(in srgb, var(--dnascape-accent) 8%, var(--dnascape-panel));
          --dnascape-sidebar-active-bg: color-mix(in srgb, var(--dnascape-text) 7%, var(--dnascape-panel));
          --dnascape-card-hover-bg: color-mix(in srgb, var(--dnascape-accent) 4%, var(--dnascape-panel));
          --dnascape-shadow: rgba(16, 24, 40, 0.04);
          --dnascape-shadow-hover: rgba(16, 24, 40, 0.07);
          --dnascape-header-height: 2.5rem;
          --dnascape-sidebar-width: 15rem;
          --dnascape-sidebar-inner-width: min(82%, 16rem);
          --dnascape-sidebar-logo-width: min(70%, 10.25rem);
          --dnascape-sidebar-logo-aspect-ratio: 1538 / 829;
          --dnascape-plotting-panel-padding: 1.2rem 1.15rem;
          --dnascape-plotting-expander-gap: 0.5rem;
          /* Dark-logo mask tweak points. The small logo keeps the same
             width-relative radius as the main logo; its y-position is scaled
             for the shorter small-logo PNG height, 930 / 829. */
          --dnascape-logo-mask-x: 48.5%;
          --dnascape-main-logo-mask-y: 2%;
          --dnascape-main-logo-mask-radius: 2.4%;
          --dnascape-main-logo-mask-edge: 2.8%;
          --dnascape-sidebar-logo-mask-x: var(--dnascape-logo-mask-x);
          --dnascape-sidebar-logo-mask-y: 2.24%;
          --dnascape-sidebar-logo-mask-radius: 2.4%;
          --dnascape-sidebar-logo-mask-edge: 2.8%;
          --dnascape-sidebar-nav-font-size: 1.06rem;
          --dnascape-sidebar-nav-gap: 0.25rem;
          --dnascape-sidebar-nav-padding-y: 0.44rem;
          --dnascape-sidebar-nav-padding-x: 0.85rem;
          --dnascape-sidebar-external-gap: 1.8rem;
          --dnascape-sidebar-external-bottom: 1.8rem;
          --dnascape-sidebar-external-min-gap: 0.85rem;
          --dnascape-sidebar-top-padding: 0rem;
          --dnascape-sidebar-toggle-inset: 0.9rem;
          --dnascape-sidebar-toggle-top: 0.55rem;
          --dnascape-sidebar-toggle-position-width: 2.25rem;
          --dnascape-main-top-padding: 0rem;
          --dnascape-mobile-main-top-padding: 1.3rem;
          --dnascape-overview-top-padding: 0rem;
          --dnascape-mobile-overview-top-padding: 1.3rem;
        }

        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
          scrollbar-gutter: stable;
        }

        :root[data-dnascape-effective-theme="light"] {
          color-scheme: light;
        }

        :root:not([data-dnascape-show-help="true"]) [data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"],
        :root:not([data-dnascape-show-help="true"]) [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
        :root:not([data-dnascape-show-help="true"]) [data-testid="stWidgetLabel"] [aria-label="Help"],
        :root:not([data-dnascape-show-help="true"]) [data-testid="stWidgetLabel"] [aria-label="help"],
        :root[data-dnascape-show-help="false"] [data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"],
        :root[data-dnascape-show-help="false"] [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
        :root[data-dnascape-show-help="false"] [data-testid="stWidgetLabel"] [aria-label="Help"],
        :root[data-dnascape-show-help="false"] [data-testid="stWidgetLabel"] [aria-label="help"] {
          display: none !important;
        }

        :root[data-dnascape-effective-theme="dark"] {
          --background-color: #151413;
          --secondary-background-color: #211f1d;
          --text-color: #f3eee8;
          --dnascape-accent: color-mix(in srgb, var(--primary-color, #b03412) 58%, #ffffff);
          --dnascape-line: rgba(243, 238, 232, 0.36);
          --dnascape-divider: rgba(243, 238, 232, 0.48);
          --dnascape-sidebar-active-bg: color-mix(in srgb, var(--dnascape-text) 7%, var(--dnascape-panel));
          --dnascape-shadow: rgba(0, 0, 0, 0.22);
          --dnascape-shadow-hover: rgba(0, 0, 0, 0.32);
          color-scheme: dark;
        }

        :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-logo img,
        :root[data-dnascape-effective-theme="dark"] .dnascape-main-logo img {
          filter: invert(1) hue-rotate(180deg) saturate(0.95) brightness(1.08);
        }

        :root[data-dnascape-effective-theme="dark"] .dnascape-main-logo img {
          -webkit-mask-image: radial-gradient(
            circle at var(--dnascape-logo-mask-x) var(--dnascape-main-logo-mask-y),
            transparent 0 var(--dnascape-main-logo-mask-radius),
            #000 var(--dnascape-main-logo-mask-edge)
          );
          -webkit-mask-repeat: no-repeat;
          -webkit-mask-size: 100% 100%;
          mask-image: radial-gradient(
            circle at var(--dnascape-logo-mask-x) var(--dnascape-main-logo-mask-y),
            transparent 0 var(--dnascape-main-logo-mask-radius),
            #000 var(--dnascape-main-logo-mask-edge)
          );
          mask-repeat: no-repeat;
          mask-size: 100% 100%;
        }

        :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-logo img {
          -webkit-mask-image: none !important;
          display: block !important;
          mask-image: none !important;
          opacity: 1 !important;
          visibility: visible !important;
        }

        :root.dnascape-theme-transition,
        :root.dnascape-theme-transition *,
        :root.dnascape-theme-transition *::before,
        :root.dnascape-theme-transition *::after {
          transition-property: background-color, border-color, box-shadow, color,
            fill, stroke, filter, opacity, caret-color, outline-color,
            text-decoration-color !important;
          transition-duration: 240ms !important;
          transition-timing-function: cubic-bezier(0.22, 1, 0.36, 1) !important;
        }

        @media (prefers-reduced-motion: reduce) {
          :root.dnascape-theme-transition,
          :root.dnascape-theme-transition *,
          :root.dnascape-theme-transition *::before,
          :root.dnascape-theme-transition *::after {
            transition-duration: 0ms !important;
          }
        }

        [data-testid="stAppViewContainer"] {
          background: var(--dnascape-bg);
          color: var(--dnascape-text);
        }

        header[data-testid="stHeader"] {
          background: color-mix(in srgb, var(--dnascape-bg) 94%, transparent);
          height: var(--dnascape-header-height) !important;
          min-height: var(--dnascape-header-height) !important;
        }

        header[data-testid="stHeader"] > div,
        header[data-testid="stHeader"] [data-testid="stToolbar"] {
          min-height: var(--dnascape-header-height) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stWidgetLabel"],
        :root[data-dnascape-effective-theme="dark"] [data-testid="stWidgetLabel"] *,
        :root[data-dnascape-effective-theme="dark"] label,
        :root[data-dnascape-effective-theme="dark"] legend {
          color: var(--dnascape-muted) !important;
        }

        :root[data-dnascape-effective-theme="dark"] input,
        :root[data-dnascape-effective-theme="dark"] textarea,
        :root[data-dnascape-effective-theme="dark"] [contenteditable="true"] {
          background: transparent !important;
          color: var(--dnascape-text) !important;
          caret-color: var(--dnascape-accent) !important;
          -webkit-text-fill-color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] input::placeholder,
        :root[data-dnascape-effective-theme="dark"] textarea::placeholder {
          color: var(--dnascape-muted) !important;
          opacity: 1 !important;
          -webkit-text-fill-color: var(--dnascape-muted) !important;
        }

        :root [data-testid="stFileUploaderDropzone"]
          [data-testid="stFileUploaderDropzoneInstructions"] span {
          color: transparent !important;
          font-size: 0 !important;
          overflow: visible !important;
          text-overflow: clip !important;
          white-space: normal !important;
          -webkit-text-fill-color: transparent !important;
        }

        :root [data-testid="stFileUploaderDropzone"]
          [data-testid="stFileUploaderDropzoneInstructions"] span::after {
          color: var(--dnascape-muted) !important;
          content: "Supported formats: TSV, CSV, TXT, GZ";
          display: inline !important;
          font-size: 0.875rem !important;
          line-height: 1.35 !important;
          -webkit-text-fill-color: var(--dnascape-muted) !important;
        }

        [data-testid="stFileUploaderDropzone"]:has(input[accept*=".npv"])
          [data-testid="stFileUploaderDropzoneInstructions"] span::after {
          content: "Supported formats: CSV, TSV, TXT, NPV, NPY, NPZ, GZ, BW, bigWig, bedGraph, BDG";
        }

        [data-baseweb="select"] svg,
        [data-baseweb="select"] [aria-hidden="true"],
        [data-baseweb="select"] > div > div:last-child {
          cursor: pointer !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="input"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="select"] > div,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="textarea"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"],
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"],
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuList"] {
          background: var(--dnascape-panel) !important;
          border-color: var(--dnascape-line) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"] > div,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"] section,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzoneInstructions"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="input"] > div,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="select"] > div > div,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="textarea"] > div,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] > div {
          background: transparent !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"] *,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="input"] *,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="select"] *,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="textarea"] *,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] *,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] *,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuList"] * {
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"] > input + div:has(+ *) {
          align-items: center !important;
          background: color-mix(in srgb, var(--dnascape-panel) 94%, var(--dnascape-bg)) !important;
          border-radius: 8px !important;
          bottom: 0.25rem !important;
          color: var(--dnascape-accent) !important;
          display: flex !important;
          justify-content: center !important;
          left: 0.25rem !important;
          position: absolute !important;
          right: 0.25rem !important;
          top: 0.25rem !important;
          z-index: 20 !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"] > input + div:has(+ *) * {
          color: var(--dnascape-accent) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChip"] {
          background: color-mix(in srgb, var(--dnascape-panel) 88%, var(--dnascape-bg)) !important;
          box-shadow: inset 0 0 0 1px var(--dnascape-line) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChipName"],
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChip"] div,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChip"] small,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChip"] span {
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChipDeleteBtn"] button,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [aria-label="Add files"] {
          background: transparent !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: var(--dnascape-muted) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [data-testid="stFileChipDeleteBtn"] button:hover,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileChips"] [aria-label="Add files"]:hover {
          background: transparent !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stFileUploaderDropzone"] button,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stNumberInput"] button,
        :root[data-dnascape-effective-theme="dark"] .stButton > button,
        :root[data-dnascape-effective-theme="dark"] button[kind]:not([data-testid="stMainMenuButton"]):not([data-testid^="stBaseButton-header"]),
        :root[data-dnascape-effective-theme="dark"] [role="button"] {
          background: color-mix(in srgb, var(--dnascape-panel) 88%, var(--dnascape-bg)) !important;
          border-color: var(--dnascape-line) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] button:hover,
        :root[data-dnascape-effective-theme="dark"] [role="button"]:hover,
        :root[data-dnascape-effective-theme="dark"] [role="menuitem"]:hover,
        :root[data-dnascape-effective-theme="dark"] [role="option"]:hover {
          background: var(--dnascape-hover-bg) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] button:disabled,
        :root[data-dnascape-effective-theme="dark"] [aria-disabled="true"],
        :root[data-dnascape-effective-theme="dark"] [disabled] {
          color: var(--dnascape-muted) !important;
          opacity: 1 !important;
          -webkit-text-fill-color: var(--dnascape-muted) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-baseweb="select"] svg,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stNumberInput"] svg,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenu"] svg,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] svg {
          color: var(--dnascape-muted) !important;
          fill: currentColor !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuButton"],
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenu"] button,
        :root[data-dnascape-effective-theme="dark"] button[data-testid^="stBaseButton-header"] {
          background: transparent !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: var(--dnascape-muted) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuButton"]:hover,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenu"] button:hover,
        :root[data-dnascape-effective-theme="dark"] button[data-testid^="stBaseButton-header"]:hover {
          background: color-mix(in srgb, var(--dnascape-text) 10%, transparent) !important;
          border-color: transparent !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button {
          background: transparent !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: var(--dnascape-muted) !important;
        }

        :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button *,
        :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button svg {
          color: inherit !important;
          fill: currentColor !important;
          opacity: 1 !important;
          stroke: currentColor !important;
          -webkit-text-fill-color: currentColor !important;
        }

        :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button:hover {
          background: color-mix(in srgb, var(--dnascape-text) 10%, transparent) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] {
          background: var(--dnascape-panel) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [role="menuitem"],
        :root[data-dnascape-effective-theme="dark"] [role="menuitemradio"],
        :root[data-dnascape-effective-theme="dark"] [role="option"],
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] a,
        :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] button {
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] > div,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [data-baseweb="menu"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [role="listbox"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] ul,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] {
          background: var(--dnascape-panel) !important;
          background-color: var(--dnascape-panel) !important;
          background-image: none !important;
          border-color: var(--dnascape-line) !important;
          color: var(--dnascape-text) !important;
          -webkit-text-fill-color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [role="option"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [role="option"] *,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] [role="option"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] [role="option"] * {
          background-color: transparent !important;
          color: var(--dnascape-text) !important;
          opacity: 1 !important;
          -webkit-text-fill-color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [role="option"]:hover,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [role="option"][aria-selected="true"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] [role="option"][data-highlighted="true"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] [role="option"]:hover,
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] [role="option"][aria-selected="true"],
        :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] [role="option"][data-highlighted="true"] {
          background: var(--dnascape-hover-bg) !important;
          background-color: var(--dnascape-hover-bg) !important;
          color: var(--dnascape-text) !important;
          -webkit-text-fill-color: var(--dnascape-text) !important;
        }

        section[data-testid="stSidebar"],
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"] {
          background: var(--dnascape-sidebar-bg);
        }

        section[data-testid="stSidebar"] {
          box-sizing: border-box;
          border-right: 0 !important;
          color: var(--dnascape-text);
          max-width: var(--dnascape-sidebar-width) !important;
          min-width: var(--dnascape-sidebar-width) !important;
          position: relative;
          width: var(--dnascape-sidebar-width) !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] {
          max-width: 0 !important;
          min-width: 0 !important;
          width: 0 !important;
        }

        section[data-testid="stSidebar"] button[aria-label*="sidebar" i],
        section[data-testid="stSidebar"] button[title*="sidebar" i],
        section[data-testid="stSidebar"] button[data-testid^="stBaseButton-header"] {
          margin-right: var(--dnascape-sidebar-toggle-inset) !important;
        }

        .dnascape-sidebar-toggle-button--collapse,
        .dnascape-sidebar-toggle-button--expand {
          position: fixed !important;
          right: auto !important;
          top: var(--dnascape-sidebar-toggle-top) !important;
          transform: none !important;
        }

        .dnascape-sidebar-toggle-button--collapse {
          left: calc(
            var(--dnascape-sidebar-width) -
            var(--dnascape-sidebar-toggle-inset) -
            var(--dnascape-sidebar-toggle-position-width)
          ) !important;
        }

        .dnascape-sidebar-toggle-button--expand {
          left: var(--dnascape-sidebar-toggle-inset) !important;
        }

        section[data-testid="stSidebar"]::after {
          background: var(--dnascape-divider);
          bottom: 0;
          content: "";
          pointer-events: none;
          position: absolute;
          right: 0;
          top: 0;
          width: 1px;
          z-index: 2;
        }

        div[data-testid="stSidebarContent"] {
          box-sizing: border-box !important;
          height: 100dvh !important;
          overflow-x: hidden !important;
          overflow-y: auto !important;
          padding-left: 0 !important;
          padding-right: 0 !important;
          scrollbar-gutter: stable !important;
        }

        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"] {
          box-sizing: border-box !important;
          max-width: 100% !important;
          min-width: 0 !important;
          padding-top: var(--dnascape-sidebar-top-padding) !important;
          width: 100% !important;
        }

        [data-testid="stSidebarResizeHandle"],
        [role="separator"],
        div[style*="col-resize"],
        div[style*="ew-resize"],
        div[style*="cursor: col-resize"],
        div[style*="cursor: ew-resize"] {
          display: none !important;
          pointer-events: none !important;
        }

        iframe[srcdoc*="dnascape-theme"],
        iframe[data-dnascape-theme-bridge="true"] {
          border: 0 !important;
          height: 0 !important;
          opacity: 0 !important;
          pointer-events: none !important;
          position: absolute !important;
          width: 0 !important;
        }

        [data-testid="stIFrame"]:has(iframe[srcdoc*="dnascape-theme"]),
        [data-testid="stElementContainer"]:has(iframe[srcdoc*="dnascape-theme"]),
        [data-dnascape-theme-bridge-wrapper="true"] {
          height: 0 !important;
          margin: 0 !important;
          min-height: 0 !important;
          overflow: hidden !important;
          padding: 0 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.dnascape-sidebar-active-style) {
          height: 0 !important;
          margin: 0 !important;
          min-height: 0 !important;
          overflow: hidden !important;
          padding: 0 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
          align-items: center;
          gap: 0.18rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"],
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] {
          margin-left: auto !important;
          margin-right: auto !important;
          margin-bottom: var(--dnascape-sidebar-nav-gap);
          width: var(--dnascape-sidebar-inner-width) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button {
          background: transparent !important;
          border: 0 !important;
          border-radius: 8px !important;
          box-sizing: border-box;
          box-shadow: none !important;
          color: var(--dnascape-text) !important;
          display: flex;
          font-size: var(--dnascape-sidebar-nav-font-size);
          font-weight: 500;
          justify-content: flex-start !important;
          line-height: 1.35;
          min-height: 0 !important;
          padding: var(--dnascape-sidebar-nav-padding-y)
            var(--dnascape-sidebar-nav-padding-x) !important;
          text-align: left !important;
          text-decoration: none !important;
          width: 100%;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a *,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button *,
        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"],
        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"] *,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[aria-current="page"],
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[aria-current="page"] * {
          color: var(--dnascape-text) !important;
        }

        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button div,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button p {
          display: block !important;
          flex: 1 1 auto !important;
          justify-content: flex-start !important;
          margin: 0 !important;
          text-align: left !important;
          width: 100% !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button:hover,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button:focus-visible {
          background: var(--dnascape-hover-bg) !important;
          color: var(--dnascape-text) !important;
          outline: none !important;
          text-decoration: none !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover *,
        section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button:hover * {
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stPageLink"] a,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button {
          background: transparent !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"],
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[aria-current="page"]) a,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[kind="primary"] {
          background: var(--dnascape-sidebar-active-bg) !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:focus-visible,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button:hover,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button:focus-visible {
          background: var(--dnascape-hover-bg) !important;
          color: var(--dnascape-text) !important;
          outline: none !important;
          text-decoration: none !important;
        }

        .dnascape-sidebar-logo {
          display: block;
          line-height: 0;
          margin: -2rem auto 2.2rem;
          width: var(--dnascape-sidebar-logo-width) !important;
        }

        .dnascape-sidebar-logo a {
          display: block;
          line-height: 0;
          text-decoration: none !important;
        }

        .dnascape-sidebar-logo img {
          display: block;
          height: auto;
          pointer-events: none;
          user-select: none;
          width: 100%;
        }

        section[data-testid="stSidebar"] .st-key-sidebar-logo-home {
          aspect-ratio: var(--dnascape-sidebar-logo-aspect-ratio);
          display: block !important;
          line-height: 0;
          margin: -2rem auto 2.2rem !important;
          overflow: hidden;
          position: relative;
          width: var(--dnascape-sidebar-logo-width) !important;
        }

        section[data-testid="stSidebar"] .st-key-sidebar-logo-home [data-testid="stElementContainer"]:has(.dnascape-sidebar-logo),
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .dnascape-sidebar-logo,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button {
          height: 100% !important;
          inset: 0;
          margin: 0 !important;
          padding: 0 !important;
          position: absolute;
          width: 100% !important;
        }

        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .dnascape-sidebar-logo img {
          height: 100%;
          object-fit: contain;
          width: 100%;
        }

        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button {
          z-index: 2;
        }

        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button .stButton,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button [data-testid="stButton"] {
          background: transparent !important;
          background-color: transparent !important;
          height: 100% !important;
          margin: 0 !important;
          padding: 0 !important;
          width: 100% !important;
        }

        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:hover,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:focus,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:focus-visible,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:active,
        section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button[disabled],
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:hover,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:focus,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:focus-visible,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button:active,
        :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"] .st-key-sidebar-logo-home .st-key-sidebar-logo-home-button button[disabled] {
          background: transparent !important;
          background-color: transparent !important;
          background-image: none !important;
          border: 0 !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: transparent !important;
          cursor: pointer;
          font-size: 0 !important;
          height: 100% !important;
          min-height: 0 !important;
          outline: none !important;
          padding: 0 !important;
          -webkit-text-fill-color: transparent !important;
          width: 100% !important;
        }

        .dnascape-main-logo {
          display: block;
          line-height: 0;
          margin: 0 auto;
          width: 90%;
        }

        .dnascape-main-logo img {
          display: block;
          height: auto;
          pointer-events: none;
          user-select: none;
          width: 100%;
        }

        .block-container {
          max-width: 1650px;
          padding-top: var(--dnascape-main-top-padding);
          padding-bottom: 4rem;
        }

        .overview-page-marker {
          display: none !important;
        }

        body:has(.overview-page-marker) .block-container {
          padding-top: var(--dnascape-overview-top-padding);
        }

        h1, h2, h3 {
          letter-spacing: 0;
          color: var(--dnascape-text);
        }

        [data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4, h5, h6) a[href^="#"],
        [data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4, h5, h6) a.anchor-link,
        :is(h1, h2, h3, h4, h5, h6) .anchor-link {
          display: none !important;
          visibility: hidden !important;
        }

        h1 {
          font-size: 3rem;
          line-height: 1.05;
          margin-bottom: 0.8rem;
        }

        h3 {
          font-size: 1.05rem;
          margin-bottom: 0.4rem;
        }

        a {
          color: var(--dnascape-accent);
          text-decoration: none;
        }

        a:hover {
          text-decoration: underline;
        }

        .hero {
          margin: 1.5rem 0 0.8rem;
          max-width: 1920px;
        }

        .hero h1 {
          margin: -1.5rem 0 0rem;
          max-width: 1920px;
          font-size: 2.1rem;
        }

        .eyebrow {
          color: var(--dnascape-accent);
          font-size: 0.82rem;
          font-weight: 700;
          letter-spacing: 0;
          margin-bottom: 0.55rem;
          text-transform: uppercase;
        }

        .lead {
          color: var(--dnascape-muted);
          font-size: 1.12rem;
          line-height: 1.7;
          margin: 0;
        }

        .muted {
          color: var(--dnascape-muted);
          font-size: 0.94rem;
          line-height: 1.6;
          min-height: 4.6rem;
        }

        .small-muted {
          color: var(--dnascape-muted);
          font-size: 0.88rem;
          line-height: 1.55;
        }

        .workflow-card {
          background: var(--dnascape-panel);
          border: 1px solid var(--dnascape-line);
          border-radius: 8px;
          box-shadow: 0 1px 2px var(--dnascape-shadow);
          color: var(--dnascape-text);
          cursor: pointer;
          display: flex;
          gap: 1rem;
          justify-content: space-between;
          min-height: 172px;
          padding: 1.2rem 1.15rem;
          pointer-events: none;
          position: relative;
          text-decoration: none !important;
          transition: background-color 140ms ease, border-color 140ms ease,
            box-shadow 140ms ease;
          z-index: 3;
        }

        .workflow-card:visited,
        .workflow-card:active,
        .workflow-card:focus,
        .workflow-card:hover {
          color: var(--dnascape-text);
          text-decoration: none !important;
        }

        .workflow-card:hover {
          background-color: var(--dnascape-card-hover-bg);
          border-color: color-mix(in srgb, var(--dnascape-accent) 28%, var(--dnascape-line));
          box-shadow: 0 5px 14px var(--dnascape-shadow-hover);
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]) {
          min-height: 172px;
          position: relative;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]) [data-testid="stElementContainer"]:has(.workflow-card) {
          margin: 0 !important;
          padding: 0 !important;
          pointer-events: none;
          position: relative;
          z-index: 3;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]) [class*="st-key-workflow-card-button-"] {
          background: transparent !important;
          background-color: transparent !important;
          background-image: none !important;
          height: 100% !important;
          inset: 0;
          margin: 0 !important;
          padding: 0 !important;
          position: absolute;
          width: 100% !important;
          z-index: 2;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] .stButton,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] [data-testid="stButton"],
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] [data-testid="stElementContainer"] {
          background: transparent !important;
          background-color: transparent !important;
          background-image: none !important;
          height: 100% !important;
          margin: 0 !important;
          padding: 0 !important;
          width: 100% !important;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:hover,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:focus,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:focus-visible,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:active,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button[disabled],
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:hover,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:focus,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:focus-visible,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button:active,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button[disabled] {
          background: transparent !important;
          background-color: transparent !important;
          background-image: none !important;
          border: 0 !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: transparent !important;
          cursor: pointer;
          font-size: 0 !important;
          height: 100% !important;
          min-height: 0 !important;
          outline: none !important;
          padding: 0 !important;
          -webkit-text-fill-color: transparent !important;
          width: 100% !important;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button *,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button *::before,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button *::after,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button *,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button *::before,
        :root[data-dnascape-effective-theme="dark"] body:has(.overview-page-marker) [class*="st-key-workflow-card-button-"] button *::after {
          background: transparent !important;
          background-color: transparent !important;
          background-image: none !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: transparent !important;
          -webkit-text-fill-color: transparent !important;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]):has(button:hover) .workflow-card,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]):has(button:focus-visible) .workflow-card {
          background-color: var(--dnascape-card-hover-bg);
          border-color: color-mix(in srgb, var(--dnascape-accent) 28%, var(--dnascape-line));
          box-shadow: 0 5px 14px var(--dnascape-shadow-hover);
        }

        .workflow-card::after {
          content: none !important;
        }

        .workflow-card-content {
          flex: 1 1 auto;
          min-width: 0;
        }

        .workflow-card-icon {
          align-items: center;
          color: var(--dnascape-accent);
          display: flex;
          flex: 0 0 auto;
          height: 3rem;
          justify-content: center;
          margin-left: auto;
          opacity: 0.88;
          width: 3rem;
          transition: color 140ms ease, opacity 140ms ease;
        }

        .workflow-card-icon svg {
          fill: none;
          height: 1.9rem;
          stroke: currentColor;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-width: 1.85;
          width: 1.9rem;
        }

        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]):has(button:hover) .workflow-card-icon,
        body:has(.overview-page-marker) [class*="st-key-workflow-card-"]:not([class*="st-key-workflow-card-button-"]):has(button:focus-visible) .workflow-card-icon {
          opacity: 1;
        }

        .workflow-card-title {
          color: var(--dnascape-text);
          font-size: 1.05rem;
          font-weight: 650;
          line-height: 1.35;
          margin: 0 0 0.55rem;
          text-decoration: none !important;
        }

        .workflow-card-copy {
          color: var(--dnascape-muted);
          font-size: 0.94rem;
          line-height: 1.6;
          margin: 0;
          text-decoration: none !important;
        }

        .workflow-card *,
        .workflow-card:hover *,
        .workflow-card:focus * {
          text-decoration: none !important;
        }

        .workflow-grid-row-spacer {
          height: 1rem;
        }

        .sidebar-external-links {
          align-items: center;
          bottom: var(--dnascape-sidebar-external-bottom);
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          justify-content: center;
          left: calc(var(--dnascape-sidebar-width) / 2);
          position: fixed;
          transform: translateX(-50%);
          width: min(
            calc(var(--dnascape-sidebar-width) * 0.82),
            16rem
          );
          z-index: 3;
        }

        .sidebar-external-icons {
          align-items: center;
          display: flex;
          gap: 1rem;
          justify-content: center;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-external-links {
          display: none !important;
        }

        :root.dnascape-sidebar-footer-hidden .sidebar-external-links {
          display: none !important;
        }

        .sidebar-external-links a {
          align-items: center;
          box-sizing: border-box;
          color: var(--dnascape-muted) !important;
          display: flex;
          height: 2rem;
          justify-content: center;
          padding: 0;
          text-decoration: none;
          transition: color 120ms ease, transform 120ms ease;
          width: 2rem;
        }

        .sidebar-external-links svg {
          flex: 0 0 auto;
          height: 1.12rem;
          width: 1.12rem;
        }

        .sidebar-external-links a:hover {
          color: var(--dnascape-accent) !important;
          text-decoration: none;
          transform: translateY(-0.05rem);
        }

        .sidebar-app-version {
          color: var(--dnascape-muted);
          font-size: 0.68rem;
          font-weight: 500;
          line-height: 1;
          opacity: 0.72;
          text-align: center;
        }

        .workflow-line {
          align-items: center;
          color: var(--dnascape-muted);
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin: 1.6rem 0 2rem;
        }

        .workflow-line span {
          background: var(--dnascape-panel);
          border: 1px solid var(--dnascape-line);
          border-radius: 999px;
          font-size: 0.84rem;
          padding: 0.42rem 0.7rem;
        }

        .workflow-line span:not(:last-child)::after {
          content: "";
        }

        .footer-note {
          border-top: 1px solid var(--dnascape-line);
          color: var(--dnascape-muted);
          font-size: 0.86rem;
          line-height: 1.6;
          margin-top: 2.4rem;
          padding-top: 1rem;
        }

        .footer-note a {
          color: var(--dnascape-accent) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--dnascape-panel);
          border-color: var(--dnascape-line) !important;
          border-radius: 8px !important;
          box-shadow: 0 1px 2px var(--dnascape-shadow);
        }

        :root[data-dnascape-effective-theme="dark"] div[data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid var(--dnascape-line) !important;
          box-shadow:
            0 1px 2px var(--dnascape-shadow),
            inset 0 0 0 1px rgba(243, 238, 232, 0.08);
        }

        .plotting-page-marker {
          display: none !important;
        }

        body:has(.plotting-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--dnascape-panel) !important;
          border: 1px solid var(--dnascape-line) !important;
          border-radius: 8px !important;
          box-shadow: 0 1px 2px var(--dnascape-shadow) !important;
          padding: var(--dnascape-plotting-panel-padding) !important;
        }

        body:has(.plotting-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
          padding: 0 !important;
        }

        body:has(.plotting-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
          gap: 0.5rem !important;
        }

        .plotting-panel-title {
          color: var(--dnascape-text);
          font-size: 1.05rem;
          font-weight: 650;
          line-height: 1.35;
          margin: 0 0 0.55rem;
        }

        .plotting-panel-title--spaced {
          margin-top: 1.45rem;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] {
          border: 0 !important;
          margin: 0 !important;
        }

        body:has(.plotting-page-marker) [data-testid="stElementContainer"]:has([data-testid="stExpander"]) {
          margin-bottom: var(--dnascape-plotting-expander-gap) !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] details {
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] summary {
          align-items: center;
          border-radius: 0.6rem;
          color: var(--dnascape-text) !important;
          cursor: pointer;
          display: flex;
          font-size: 0.96rem;
          font-weight: 680;
          line-height: 1.35;
          padding: 0.24rem 0.1rem !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] summary p,
        body:has(.plotting-page-marker) [data-testid="stExpander"] summary span {
          font-weight: 680 !important;
          margin: 0 !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] summary:hover {
          color: var(--dnascape-accent) !important;
          background: color-mix(in srgb, var(--dnascape-accent) 4%, transparent);
        }

        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary,
        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary > div,
        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary > div > div,
        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"],
        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary [data-baseweb],
        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary [role="button"] {
          background: transparent !important;
          border-color: transparent !important;
          box-shadow: none !important;
          color: var(--dnascape-text) !important;
        }

        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] details[open] > summary,
        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) [data-testid="stExpander"] summary:hover {
          background: var(--dnascape-hover-bg) !important;
          color: var(--dnascape-accent) !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] summary svg {
          color: var(--dnascape-muted) !important;
          fill: currentColor !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpander"] details[open] > summary svg,
        body:has(.plotting-page-marker) [data-testid="stExpander"] summary:hover svg {
          color: var(--dnascape-accent) !important;
        }

        body:has(.plotting-page-marker) [data-testid="stExpanderDetails"] {
          padding: 0.5rem 0 0.1rem !important;
        }

        :root[data-dnascape-effective-theme="dark"] body:has(.plotting-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: rgba(243, 238, 232, 0.52) !important;
          box-shadow:
            0 1px 2px var(--dnascape-shadow),
            inset 0 0 0 1px rgba(243, 238, 232, 0.16) !important;
        }

        .plot-preview-frame {
          border-radius: 8px;
          margin-bottom: 0.95rem;
          position: relative;
          width: 100%;
        }

        .plot-preview-expand-toggle {
          opacity: 0;
          pointer-events: none;
          position: absolute;
        }

        .plot-preview-actions {
          display: flex;
          gap: 0.45rem;
          opacity: 0;
          pointer-events: none;
          position: absolute;
          right: 0.65rem;
          top: 0.65rem;
          transition: opacity 140ms ease;
          z-index: 3;
        }

        .plot-preview-frame:hover .plot-preview-actions,
        .st-key-plot-preview-shell:hover .plot-preview-actions,
        .st-key-map-preview-shell:hover .plot-preview-actions,
        .st-key-simulation-preview-shell:hover .plot-preview-actions,
        .plot-preview-actions:focus-within {
          opacity: 1;
          pointer-events: auto;
        }

        .plot-preview-action {
          align-items: center;
          backdrop-filter: blur(8px);
          background: color-mix(in srgb, var(--dnascape-bg) 82%, transparent);
          border: 1px solid var(--dnascape-line);
          border-radius: 0.7rem;
          box-shadow: 0 1px 2px var(--dnascape-shadow);
          color: var(--dnascape-text);
          cursor: default;
          display: flex;
          height: 2rem;
          justify-content: center;
          padding: 0;
          width: 2rem;
        }

        .plot-preview-expand-button {
          cursor: pointer;
        }

        .plot-preview-clear-placeholder {
          color: var(--dnascape-text) !important;
          pointer-events: none;
        }

        .plot-preview-clear-placeholder svg {
          display: block;
          height: 1rem;
          width: 1rem;
        }

        .st-key-plot-preview-shell,
        .st-key-map-preview-shell,
        .st-key-simulation-preview-shell {
          position: relative;
        }

        .st-key-plot-preview-clear,
        .st-key-map-preview-clear,
        .st-key-simulation-preview-clear {
          opacity: 0;
          pointer-events: none;
          position: absolute;
          right: 3.1rem;
          top: 0.65rem;
          transition: opacity 140ms ease;
          z-index: 4;
        }

        .st-key-plot-preview-shell:hover .st-key-plot-preview-clear,
        .st-key-map-preview-shell:hover .st-key-map-preview-clear,
        .st-key-simulation-preview-shell:hover .st-key-simulation-preview-clear,
        .st-key-plot-preview-clear:focus-within,
        .st-key-map-preview-clear:focus-within,
        .st-key-simulation-preview-clear:focus-within {
          opacity: 1;
          pointer-events: auto;
        }

        .st-key-plot-preview-clear .stButton,
        .st-key-map-preview-clear .stButton,
        .st-key-simulation-preview-clear .stButton {
          margin: 0 !important;
        }

        .st-key-plot-preview-clear button,
        .st-key-map-preview-clear button,
        .st-key-simulation-preview-clear button {
          align-items: center !important;
          background: transparent !important;
          border: 0 !important;
          border-radius: 0.7rem !important;
          cursor: pointer !important;
          display: flex !important;
          height: 2rem !important;
          justify-content: center !important;
          gap: 0 !important;
          line-height: 1 !important;
          min-height: 2rem !important;
          opacity: 0 !important;
          padding: 0 !important;
          width: 2rem !important;
        }

        .st-key-plot-preview-clear button > div,
        .st-key-map-preview-clear button > div,
        .st-key-simulation-preview-clear button > div {
          align-items: center !important;
          display: flex !important;
          gap: 0 !important;
          height: 100% !important;
          justify-content: center !important;
          margin: 0 !important;
          padding: 0 !important;
          width: 100% !important;
        }

        .st-key-plot-preview-clear button:hover,
        .st-key-map-preview-clear button:hover,
        .st-key-simulation-preview-clear button:hover {
          background: transparent !important;
          border-color: transparent !important;
        }

        .st-key-plot-preview-shell:has(.st-key-plot-preview-clear:hover) .plot-preview-clear-placeholder,
        .st-key-map-preview-shell:has(.st-key-map-preview-clear:hover) .plot-preview-clear-placeholder,
        .st-key-simulation-preview-shell:has(.st-key-simulation-preview-clear:hover) .plot-preview-clear-placeholder {
          background: var(--dnascape-hover-bg) !important;
          border-color: color-mix(in srgb, var(--dnascape-accent) 24%, var(--dnascape-line)) !important;
        }

        .st-key-plot-preview-clear button p,
        .st-key-map-preview-clear button p,
        .st-key-simulation-preview-clear button p {
          display: none !important;
        }

        .st-key-plot-preview-clear button [data-testid="stIconMaterial"],
        .st-key-map-preview-clear button [data-testid="stIconMaterial"],
        .st-key-simulation-preview-clear button [data-testid="stIconMaterial"] {
          align-items: center !important;
          display: flex !important;
          font-size: 1rem !important;
          height: 1rem !important;
          justify-content: center !important;
          line-height: 1 !important;
          margin: 0 !important;
          padding: 0 !important;
          width: 1rem !important;
        }

        .st-key-plot-preview-shell:has(.plot-preview-expand-toggle:checked) .st-key-plot-preview-clear,
        .st-key-map-preview-shell:has(.plot-preview-expand-toggle:checked) .st-key-map-preview-clear,
        .st-key-simulation-preview-shell:has(.plot-preview-expand-toggle:checked) .st-key-simulation-preview-clear {
          opacity: 1;
          pointer-events: auto;
          position: fixed;
          right: calc(clamp(1rem, 3vw, 3rem) + 3.45rem);
          top: calc(clamp(1rem, 3vw, 3rem) + 1rem);
          z-index: 1000000;
        }

        .plot-download-menu {
          position: relative;
        }

        .plot-download-menu button {
          cursor: pointer;
        }

        .plot-download-menu-panel {
          background: var(--dnascape-panel);
          border: 1px solid var(--dnascape-line);
          border-radius: 8px;
          box-shadow: 0 8px 22px var(--dnascape-shadow-hover);
          min-width: 4.5rem;
          opacity: 0;
          overflow: hidden;
          padding: 0.2rem;
          pointer-events: none;
          position: absolute;
          right: 0;
          top: calc(100% + 0.35rem);
          transform: translateY(-0.15rem);
          transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
          visibility: hidden;
          z-index: 5;
        }

        .plot-download-menu:focus-within .plot-download-menu-panel {
          opacity: 1;
          pointer-events: auto;
          transform: translateY(0);
          visibility: visible;
        }

        .plot-download-menu-panel a {
          border-radius: 6px;
          color: var(--dnascape-text) !important;
          display: block;
          font-size: 0.86rem;
          line-height: 1.2;
          padding: 0.45rem 0.65rem;
          text-decoration: none !important;
        }

        .plot-download-menu-panel a:hover {
          background: var(--dnascape-hover-bg);
          color: var(--dnascape-text) !important;
        }

        .plot-preview-action:hover {
          background: var(--dnascape-hover-bg);
          border-color: color-mix(in srgb, var(--dnascape-accent) 24%, var(--dnascape-line));
        }

        .plot-preview-action svg {
          display: block;
          height: 1rem;
          width: 1rem;
        }

        .plot-preview-image {
          display: block;
          height: auto;
          margin: 0 auto;
          max-width: 100%;
          width: 100%;
        }

        .plot-preview-frame:has(.plot-preview-expand-toggle:checked) {
          align-items: center;
          background: var(--dnascape-bg);
          box-shadow: 0 18px 60px var(--dnascape-shadow-hover);
          display: flex;
          inset: clamp(1rem, 3vw, 3rem);
          justify-content: center;
          margin: 0 !important;
          overflow: auto;
          padding: 1.25rem;
          position: fixed;
          width: auto;
          z-index: 999999;
        }

        .plot-preview-frame:has(.plot-preview-expand-toggle:checked) .plot-preview-actions {
          opacity: 1;
          pointer-events: auto;
          right: 1rem;
          top: 1rem;
        }

        .plot-preview-frame:has(.plot-preview-expand-toggle:checked) .plot-preview-image {
          max-height: calc(100vh - 5.5rem);
          max-width: calc(100vw - 5.5rem);
          object-fit: contain;
          width: auto;
        }

        :root[data-dnascape-effective-theme="dark"] .plot-preview-frame {
          background: #ffffff;
          box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.08);
          overflow: hidden;
          padding: 0.9rem;
        }

        :root[data-dnascape-effective-theme="dark"] .plot-preview-frame:has(.plot-preview-expand-toggle:checked) {
          box-shadow: 0 18px 60px var(--dnascape-shadow-hover);
          overflow: auto;
          padding: 1.25rem;
        }

        .stButton > button,
        div[data-testid="stPageLink"] a {
          border-radius: 8px;
          font-weight: 550;
        }

        .fallback-link {
          display: inline-flex;
          font-weight: 600;
          margin-top: 0.25rem;
        }

        @media (max-width: 700px) {
          .block-container {
            padding-top: var(--dnascape-mobile-main-top-padding);
          }

          body:has(.overview-page-marker) .block-container {
            padding-top: var(--dnascape-mobile-overview-top-padding);
          }

          h1 {
            font-size: 2.35rem;
          }

          .lead {
            font-size: 1rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle() -> None:
    """Add the theme control to Streamlit's three-dot menu."""
    st.iframe(
        f"""
        <script>
        (() => {{
          const win = window.parent;
          const doc = win.document;
          const root = doc.documentElement;
          const storageKey = "dnascape-theme";
          const helpStorageKey = "dnascape-show-help";
          const styleId = "dnascape-theme-menu-style";
          const styleVersion = "theme-menu-critical-paint-19";
          const wrapperId = "dnascape-theme-menu-item-wrapper";
          const itemId = "dnascape-theme-menu-item";
          const helpWrapperId = "dnascape-help-menu-item-wrapper";
          const helpItemId = "dnascape-help-menu-item";
          const selectboxToggleVersion = "selectbox-toggle-collapse-3";
          const themes = {list(THEMES)!r};
          const icons = {json.dumps(THEME_ICONS)};
          const helpIcons = {{
            true: `
              <svg class="dnascape-theme-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <circle cx="12" cy="12" r="8.25" fill="none" stroke="currentColor" stroke-width="1.8"/>
                <path d="M9.7 9.15a2.45 2.45 0 0 1 4.75.85c0 1.85-2.45 2.05-2.45 3.55" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
                <circle cx="12" cy="17" r="1" fill="currentColor"/>
              </svg>
            `,
            false: `
              <svg class="dnascape-theme-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <circle cx="12" cy="12" r="8.25" fill="none" stroke="currentColor" stroke-width="1.8"/>
                <path d="M9.7 9.15a2.45 2.45 0 0 1 4.75.85c0 1.85-2.45 2.05-2.45 3.55" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
                <circle cx="12" cy="17" r="1" fill="currentColor"/>
                <path d="M4.75 19.25 19.25 4.75" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
              </svg>
            `,
          }};
          const labels = {THEME_LABELS!r};
          const prefersDark = win.matchMedia("(prefers-color-scheme: dark)");

          function hideBridgeFrame() {{
            const frame = window.frameElement;
            if (!frame) {{
              return;
            }}

            frame.dataset.dnascapeThemeBridge = "true";
            frame.setAttribute("aria-hidden", "true");
            frame.setAttribute("tabindex", "-1");
            Object.assign(frame.style, {{
              border: "0",
              height: "0",
              opacity: "0",
              pointerEvents: "none",
              position: "absolute",
              width: "0",
            }});

            let node = frame.parentElement;
            let depth = 0;
            while (node && node !== doc.body && depth < 4) {{
              const testId = node.getAttribute("data-testid") || "";
              const isIframeWrapper = /stIFrame|stElementContainer/i.test(testId);
              const isSharedLayout = /stVerticalBlock|stAppViewContainer|stMain|stSidebar/i.test(testId);
              if (!isIframeWrapper && (isSharedLayout || node.children.length > 1)) {{
                break;
              }}

              node.dataset.dnascapeThemeBridgeWrapper = "true";
              Object.assign(node.style, {{
                height: "0",
                margin: "0",
                minHeight: "0",
                overflow: "hidden",
                padding: "0",
              }});
              node = node.parentElement;
              depth += 1;
            }}
          }}

          function normalize(theme) {{
            return themes.includes(theme) ? theme : "light";
          }}

          function effectiveTheme(theme) {{
            return theme === "system" ? (prefersDark.matches ? "dark" : "light") : theme;
          }}

          function nextTheme(theme) {{
            const index = themes.indexOf(normalize(theme));
            return themes[(index + 1) % themes.length];
          }}

          function storedTheme() {{
            try {{
              return normalize(win.localStorage.getItem(storageKey));
            }} catch (error) {{
              return "light";
            }}
          }}

          function setStoredTheme(theme) {{
            try {{
              win.localStorage.setItem(storageKey, theme);
            }} catch (error) {{
              /* Storage can be unavailable in hardened browser contexts. */
            }}
          }}

          function storedHelp() {{
            try {{
              return win.localStorage.getItem(helpStorageKey) === "true";
            }} catch (error) {{
              return false;
            }}
          }}

          function setStoredHelp(showHelp) {{
            try {{
              win.localStorage.setItem(helpStorageKey, showHelp ? "true" : "false");
            }} catch (error) {{
              /* Storage can be unavailable in hardened browser contexts. */
            }}
          }}

          function injectStyle() {{
            const existing = doc.getElementById(styleId);
            if (existing && existing.dataset.dnascapeVersion === styleVersion) {{
              return;
            }}

            const style = existing || doc.createElement("style");
            style.id = styleId;
            style.dataset.dnascapeVersion = styleVersion;
            style.textContent = `
              :root {{
                --dnascape-sidebar-width: 15rem;
              }}

              section[data-testid="stSidebar"] {{
                box-sizing: border-box !important;
                max-width: var(--dnascape-sidebar-width) !important;
                min-width: var(--dnascape-sidebar-width) !important;
                width: var(--dnascape-sidebar-width) !important;
              }}

              section[data-testid="stSidebar"][aria-expanded="false"] {{
                max-width: 0 !important;
                min-width: 0 !important;
                width: 0 !important;
              }}

              div[data-testid="stSidebarContent"],
              div[data-testid="stSidebarUserContent"] {{
                box-sizing: border-box !important;
                max-width: 100% !important;
                min-width: 0 !important;
                width: 100% !important;
              }}

              [data-testid="stSidebarResizeHandle"],
              [role="separator"],
              div[style*="col-resize"],
              div[style*="ew-resize"],
              div[style*="cursor: col-resize"],
              div[style*="cursor: ew-resize"] {{
                display: none !important;
                pointer-events: none !important;
              }}

              :root[data-dnascape-effective-theme="dark"] {{
                --background-color: #151413;
                --secondary-background-color: #211f1d;
                --text-color: #f3eee8;
                --dnascape-bg: #151413;
                --dnascape-panel: #211f1d;
                --dnascape-text: #f3eee8;
                --dnascape-sidebar-bg: #1b1918;
                --dnascape-line: rgba(243, 238, 232, 0.36);
                --dnascape-divider: rgba(243, 238, 232, 0.48);
                color-scheme: dark;
              }}

              :root[data-dnascape-effective-theme="dark"],
              :root[data-dnascape-effective-theme="dark"] body,
              :root[data-dnascape-effective-theme="dark"] #root,
              :root[data-dnascape-effective-theme="dark"] .stApp,
              :root[data-dnascape-effective-theme="dark"] [data-testid="stApp"],
              :root[data-dnascape-effective-theme="dark"] [data-testid="stAppViewContainer"],
              :root[data-dnascape-effective-theme="dark"] [data-testid="stMain"] {{
                background: #151413 !important;
                color: #f3eee8 !important;
              }}

              :root[data-dnascape-effective-theme="dark"] header[data-testid="stHeader"] {{
                background: rgba(21, 20, 19, 0.94) !important;
              }}

              :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button {{
                background: transparent !important;
                border-color: transparent !important;
                box-shadow: none !important;
                color: rgba(243, 238, 232, 0.72) !important;
              }}

              :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button *,
              :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button svg {{
                color: inherit !important;
                fill: currentColor !important;
                opacity: 1 !important;
                stroke: currentColor !important;
                -webkit-text-fill-color: currentColor !important;
              }}

              :root[data-dnascape-effective-theme="dark"] .dnascape-sidebar-toggle-button:hover {{
                background: rgba(250, 250, 250, 0.10) !important;
                color: #f3eee8 !important;
              }}

              :root[data-dnascape-effective-theme="dark"] section[data-testid="stSidebar"],
              :root[data-dnascape-effective-theme="dark"] div[data-testid="stSidebarContent"],
              :root[data-dnascape-effective-theme="dark"] div[data-testid="stSidebarUserContent"] {{
                background: #1b1918 !important;
                color: #f3eee8 !important;
              }}

              :root[data-dnascape-effective-theme="dark"] [data-testid="stDecoration"],
              :root[data-dnascape-effective-theme="dark"] [data-testid="stStatusWidget"] {{
                background: transparent !important;
              }}

              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"],
              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuList"],
              :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"],
              :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] {{
                background: #211f1d !important;
                border-color: rgba(243, 238, 232, 0.14) !important;
                color: #f3eee8 !important;
              }}

              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] *,
              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuList"] *,
              :root[data-dnascape-effective-theme="dark"] [data-baseweb="popover"] *,
              :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] * {{
                color: #f3eee8 !important;
                fill: currentColor !important;
              }}

              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] [role="menuitem"]:hover,
              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] button:hover,
              :root[data-dnascape-effective-theme="dark"] [data-testid="stMainMenuPopover"] a:hover,
              :root[data-dnascape-effective-theme="dark"] [data-baseweb="menu"] [role="option"]:hover {{
                background: rgba(250, 250, 250, 0.10) !important;
                color: #f3eee8 !important;
              }}

              iframe[srcdoc*="dnascape-theme"],
              iframe[data-dnascape-theme-bridge="true"] {{
                border: 0 !important;
                height: 0 !important;
                opacity: 0 !important;
                pointer-events: none !important;
                position: absolute !important;
                width: 0 !important;
              }}

              [data-testid="stIFrame"]:has(iframe[srcdoc*="dnascape-theme"]),
              [data-testid="stElementContainer"]:has(iframe[srcdoc*="dnascape-theme"]),
              [data-dnascape-theme-bridge-wrapper="true"] {{
                height: 0 !important;
                margin: 0 !important;
                min-height: 0 !important;
                overflow: hidden !important;
                padding: 0 !important;
              }}

              #${{wrapperId}},
              #${{helpWrapperId}} {{
                display: block;
                list-style: none;
                margin: 0;
                padding: 0;
              }}

              #${{itemId}},
              #${{helpItemId}} {{
                align-items: center;
                box-sizing: border-box;
                cursor: pointer;
                display: flex;
                gap: 0.3rem;
                justify-content: flex-start;
                width: 100%;
              }}

              #${{itemId}}:not(.dnascape-native-menu-item),
              #${{helpItemId}}:not(.dnascape-native-menu-item) {{
                appearance: none;
                background: transparent;
                border: 0;
                color: inherit;
                font: inherit;
                line-height: 1.35;
                min-height: 2.25rem;
                padding: 0.48rem 0.85rem;
                text-align: left;
              }}

              #${{wrapperId}}:hover #${{itemId}},
              #${{wrapperId}}:focus-within #${{itemId}},
              #${{helpWrapperId}}:hover #${{helpItemId}},
              #${{helpWrapperId}}:focus-within #${{helpItemId}},
              #${{itemId}}:hover,
              #${{itemId}}:focus-visible,
              #${{helpItemId}}:hover,
              #${{helpItemId}}:focus-visible {{
                background: rgba(250, 250, 250, 0.10) !important;
                border-radius: 0.5rem !important;
                outline: none !important;
              }}

              :root[data-dnascape-effective-theme="dark"] #${{wrapperId}}:hover #${{itemId}},
              :root[data-dnascape-effective-theme="dark"] #${{wrapperId}}:focus-within #${{itemId}},
              :root[data-dnascape-effective-theme="dark"] #${{helpWrapperId}}:hover #${{helpItemId}},
              :root[data-dnascape-effective-theme="dark"] #${{helpWrapperId}}:focus-within #${{helpItemId}},
              :root[data-dnascape-effective-theme="dark"] #${{itemId}}:hover,
              :root[data-dnascape-effective-theme="dark"] #${{itemId}}:focus-visible,
              :root[data-dnascape-effective-theme="dark"] #${{helpItemId}}:hover,
              :root[data-dnascape-effective-theme="dark"] #${{helpItemId}}:focus-visible {{
                background: rgba(250, 250, 250, 0.10) !important;
              }}

              #${{itemId}} .dnascape-theme-menu-icon,
              #${{helpItemId}} .dnascape-theme-menu-icon {{
                align-items: center;
                color: currentColor;
                display: inline-flex;
                flex: 0 0 auto;
                justify-content: center;
                line-height: 0;
                pointer-events: none;
              }}

              #${{itemId}} .dnascape-theme-menu-icon svg,
              #${{helpItemId}} .dnascape-theme-menu-icon svg {{
                display: block;
                height: 1.08rem;
                pointer-events: none;
                width: 1.08rem;
              }}

              #${{itemId}} .dnascape-theme-menu-icon *,
              #${{helpItemId}} .dnascape-theme-menu-icon * {{
                pointer-events: none;
              }}

              #${{itemId}} .dnascape-theme-menu-label,
              #${{helpItemId}} .dnascape-theme-menu-label {{
                flex: 0 1 auto;
                overflow: hidden;
                pointer-events: none;
                text-overflow: ellipsis;
                white-space: nowrap;
              }}
            `;
            if (!existing) {{
              doc.head.appendChild(style);
            }}
          }}

          function removeHeaderExperiment() {{
            doc
              .querySelectorAll("#dnascape-theme-toggle, .dnascape-theme-toggle, [data-testid='dnascapeThemeToggle']")
              .forEach((node) => node.remove());

            const oldStyle = doc.getElementById("dnascape-theme-toggle-style");
            if (oldStyle) {{
              oldStyle.remove();
            }}
          }}

          function menuList() {{
            const popover = doc.querySelector(
              '[data-testid="stMainMenuPopover"], .stMainMenuPopover'
            );
            if (!popover) {{
              return null;
            }}

            return (
              popover.querySelector('[data-testid="stMainMenuList"][role="menu"]') ||
              popover.querySelector('[data-testid="stMainMenuList"]') ||
              popover.querySelector('[role="menu"]') ||
              popover
            );
          }}

          function directMenuChild(node, list) {{
            let current = node;
            while (current && current.parentElement !== list) {{
              current = current.parentElement;
            }}
            return current || null;
          }}

          function nativeMenuItem(list) {{
            const candidates = Array
              .from(list.querySelectorAll('a, button, [role="menuitem"], [role="menuitemradio"]'))
              .filter((node) => (
                node.id !== itemId &&
                node.id !== helpItemId &&
                !node.closest(`#${{wrapperId}}`) &&
                !node.closest(`#${{helpWrapperId}}`)
              ));

            return (
              candidates.find((node) => /about|report a bug|get help/i.test(node.textContent || "")) ||
              candidates[0] ||
              null
            );
          }}

          function compactText(node) {{
            return (node.textContent || "").replace(/\\s+/g, " ").trim();
          }}

          function hideNode(node) {{
            if (!node || node.closest(`#${{wrapperId}}`) || node.closest(`#${{helpWrapperId}}`)) {{
              return;
            }}
            node.dataset.dnascapeHiddenThemeControl = "true";
            node.style.display = "none";
          }}

          function directPopoverChild(node, popover) {{
            let current = node;
            while (current && current.parentElement !== popover) {{
              current = current.parentElement;
            }}
            return current || node;
          }}

          function hideBuiltInThemeControls() {{
            const popover = doc.querySelector(
              '[data-testid="stMainMenuPopover"], .stMainMenuPopover'
            );
            if (!popover) {{
              return;
            }}

            const themeChoiceLabels = new Set(["System", "Light", "Dark"]);
            const themeChoices = Array
              .from(popover.querySelectorAll('button, [role="button"], [role="option"], [role="menuitemradio"], label'))
              .filter((node) => (
                !node.closest(`#${{wrapperId}}`) &&
                !node.closest(`#${{helpWrapperId}}`) &&
                themeChoiceLabels.has(compactText(node))
              ));

            if (themeChoices.length >= 3) {{
              let common = themeChoices[0].parentElement;
              while (
                common &&
                common !== popover &&
                !themeChoices.every((node) => common.contains(node))
              ) {{
                common = common.parentElement;
              }}

              if (common && common !== popover) {{
                hideNode(common);
              }} else {{
                themeChoices.forEach((node) => hideNode(directPopoverChild(node, popover)));
              }}
            }}

            Array
              .from(popover.querySelectorAll("div, section, ul, li"))
              .filter((node) => !node.closest(`#${{wrapperId}}`) && !node.closest(`#${{helpWrapperId}}`))
              .filter((node) => compactText(node).replace(/\\s+/g, "") === "SystemLightDark")
              .forEach((node) => hideNode(node));

            Array
              .from(popover.querySelectorAll('[role="menuitem"], [role="option"], [role="menuitemradio"], button, a'))
              .filter((node) => !node.closest(`#${{wrapperId}}`) && !node.closest(`#${{helpWrapperId}}`))
              .forEach((node) => {{
                const text = compactText(node);
                if (/^(settings|theme|appearance)$/i.test(text) || /choose app theme/i.test(text)) {{
                  hideNode(directPopoverChild(node, popover));
                }}
              }});
          }}

          function syncNativeMenuStyle(list, wrapper, item) {{
            const reference = nativeMenuItem(list);
            if (!reference) {{
              return;
            }}

            const referenceWrapper = directMenuChild(reference, list);
            const wrapperClass = referenceWrapper && referenceWrapper.getAttribute("class");
            if (wrapperClass) {{
              wrapper.setAttribute("class", wrapperClass);
            }} else {{
              wrapper.removeAttribute("class");
            }}

            const itemClass = reference.getAttribute("class");
            if (itemClass) {{
              item.setAttribute("class", `${{itemClass}} dnascape-native-menu-item`);
            }} else {{
              item.setAttribute("class", "dnascape-native-menu-item");
            }}

            const itemStyle = reference.getAttribute("style");
            if (itemStyle) {{
              item.setAttribute("style", itemStyle);
            }} else {{
              item.removeAttribute("style");
            }}

            item.style.alignItems = "center";
            item.style.display = "flex";
            item.style.gap = "0.3rem";
            item.style.justifyContent = "flex-start";
          }}

          function updateMenuItem(theme) {{
            const item = doc.getElementById(itemId);
            if (!item) {{
              return;
            }}

            const label = `Theme: ${{labels[theme]}}`;
            item.setAttribute("aria-label", label);
            item.removeAttribute("title");

            const iconNode = item.querySelector(".dnascape-theme-menu-icon");
            if (iconNode) {{
              iconNode.innerHTML = icons[theme];
            }}

            const labelNode = item.querySelector(".dnascape-theme-menu-label");
            if (labelNode) {{
              labelNode.textContent = label;
              if (iconNode && labelNode.nextElementSibling !== iconNode) {{
                item.insertBefore(labelNode, iconNode);
              }}
            }} else {{
              item.textContent = label;
            }}
          }}

          function updateHelpMenuItem(showHelp) {{
            const item = doc.getElementById(helpItemId);
            if (!item) {{
              return;
            }}

            const label = showHelp ? "Hide help" : "Show help";
            const stateLabel = showHelp ? "On" : "Off";
            item.setAttribute("aria-label", `${{label}}: ${{stateLabel}}`);
            item.setAttribute("aria-pressed", showHelp ? "true" : "false");
            item.removeAttribute("title");

            const iconNode = item.querySelector(".dnascape-theme-menu-icon");
            if (iconNode) {{
              iconNode.innerHTML = helpIcons[showHelp ? "true" : "false"];
            }}

            const labelNode = item.querySelector(".dnascape-theme-menu-label");
            if (labelNode) {{
              labelNode.textContent = label;
              if (iconNode && labelNode.nextElementSibling !== iconNode) {{
                item.insertBefore(labelNode, iconNode);
              }}
            }} else {{
              item.textContent = label;
            }}
          }}

          function applyHelp(showHelp) {{
            const selected = Boolean(showHelp);
            root.setAttribute("data-dnascape-show-help", selected ? "true" : "false");
            setStoredHelp(selected);
            updateHelpMenuItem(selected);
          }}

          function stopMenuPointer(event) {{
            event.stopPropagation();
          }}

          function activateMenuItem(event) {{
            event.preventDefault();
            event.stopImmediatePropagation();
            const current = normalize(root.getAttribute("data-dnascape-theme") || storedTheme());
            applyTheme(nextTheme(current), true);
          }}

          function activateHelpMenuItem(event) {{
            event.preventDefault();
            event.stopImmediatePropagation();
            applyHelp(root.getAttribute("data-dnascape-show-help") !== "true");
          }}

          function bindMenuItem(item) {{
            if (!item || item.dataset.dnascapeThemeItemVersion === styleVersion) {{
              return;
            }}

            item.dataset.dnascapeThemeItemVersion = styleVersion;
            item.addEventListener("pointerdown", stopMenuPointer, true);
            item.addEventListener("mousedown", stopMenuPointer, true);
            item.addEventListener("click", activateMenuItem, true);
          }}

          function bindHelpMenuItem(item) {{
            if (!item || item.dataset.dnascapeHelpItemVersion === styleVersion) {{
              return;
            }}

            item.dataset.dnascapeHelpItemVersion = styleVersion;
            item.addEventListener("pointerdown", stopMenuPointer, true);
            item.addEventListener("mousedown", stopMenuPointer, true);
            item.addEventListener("click", activateHelpMenuItem, true);
          }}

          function createMenuItem() {{
            const wrapper = doc.createElement("li");
            wrapper.id = wrapperId;
            wrapper.setAttribute("role", "none");

            const item = doc.createElement("button");
            item.id = itemId;
            item.type = "button";
            item.setAttribute("role", "menuitem");
            item.innerHTML = `
              <span class="dnascape-theme-menu-label"></span>
              <span class="dnascape-theme-menu-icon" aria-hidden="true"></span>
            `;
            bindMenuItem(item);

            wrapper.appendChild(item);
            return wrapper;
          }}

          function createHelpMenuItem() {{
            const wrapper = doc.createElement("li");
            wrapper.id = helpWrapperId;
            wrapper.setAttribute("role", "none");

            const item = doc.createElement("button");
            item.id = helpItemId;
            item.type = "button";
            item.setAttribute("role", "menuitem");
            item.innerHTML = `
              <span class="dnascape-theme-menu-label"></span>
              <span class="dnascape-theme-menu-icon" aria-hidden="true"></span>
            `;
            bindHelpMenuItem(item);

            wrapper.appendChild(item);
            return wrapper;
          }}

          function ensureMenuItem() {{
            injectStyle();
            removeHeaderExperiment();

            const list = menuList();
            if (!list) {{
              return false;
            }}

            let wrapper = doc.getElementById(wrapperId);
            if (!wrapper) {{
              wrapper = createMenuItem();
            }}

            let helpWrapper = doc.getElementById(helpWrapperId);
            if (!helpWrapper) {{
              helpWrapper = createHelpMenuItem();
            }}

            bindMenuItem(wrapper.querySelector(`#${{itemId}}`));
            bindHelpMenuItem(helpWrapper.querySelector(`#${{helpItemId}}`));

            if (wrapper.parentElement !== list) {{
              list.insertBefore(wrapper, list.firstElementChild);
            }} else if (list.firstElementChild !== wrapper) {{
              list.insertBefore(wrapper, list.firstElementChild);
            }}

            if (helpWrapper.parentElement !== list) {{
              list.insertBefore(helpWrapper, wrapper.nextSibling);
            }} else if (helpWrapper.previousSibling !== wrapper) {{
              list.insertBefore(helpWrapper, wrapper.nextSibling);
            }}

            syncNativeMenuStyle(list, wrapper, wrapper.querySelector(`#${{itemId}}`));
            syncNativeMenuStyle(list, helpWrapper, helpWrapper.querySelector(`#${{helpItemId}}`));
            updateMenuItem(normalize(root.getAttribute("data-dnascape-theme") || storedTheme()));
            updateHelpMenuItem(storedHelp());
            hideBuiltInThemeControls();
            return true;
          }}

          function syncMenuSoon() {{
            [0, 25, 80, 160, 320, 650].forEach((delay) => {{
              win.setTimeout(ensureMenuItem, delay);
            }});
          }}

          function bindMenuButton() {{
            doc
              .querySelectorAll('[data-testid="stMainMenuButton"], [data-testid="stMainMenu"] button, #MainMenu button')
              .forEach((button) => {{
                if (button.dataset.dnascapeThemeBound === "true") {{
                  return;
                }}
                button.dataset.dnascapeThemeBound = "true";
                button.addEventListener("click", syncMenuSoon, true);
                button.addEventListener("keydown", syncMenuSoon, true);
              }});
          }}

          function installSelectboxToggleCollapse() {{
            if (win.__dnascapeSelectboxToggleVersion === selectboxToggleVersion) {{
              return;
            }}

            if (typeof win.__dnascapeSelectboxToggleCleanup === "function") {{
              win.__dnascapeSelectboxToggleCleanup();
            }}

            const suppressClickUntil = new WeakMap();

            function eventElement(event) {{
              return event.target instanceof win.Element ? event.target : null;
            }}

            function selectboxHit(event) {{
              const target = eventElement(event);
              if (!target) {{
                return null;
              }}

              const wrapper = target.closest('[data-testid="stSelectbox"]');
              if (!wrapper) {{
                return null;
              }}

              const control = target.closest('[data-baseweb="select"]');
              if (!control || !wrapper.contains(control)) {{
                return null;
              }}

              return {{ wrapper, control }};
            }}

            function selectboxReportsExpanded(hit) {{
              return Boolean(
                hit.control.matches('[aria-expanded="true"]') ||
                hit.control.querySelector('[aria-expanded="true"]') ||
                hit.wrapper.querySelector('[aria-expanded="true"]')
              );
            }}

            function selectboxIsExpanded(hit) {{
              return Boolean(
                selectboxReportsExpanded(hit) ||
                doc.querySelector('[data-baseweb="popover"] [role="option"], [data-baseweb="menu"] [role="option"]')
              );
            }}

            function sendEscapeToClose(hit) {{
              const keyboardTargets = [
                hit.control.querySelector("input") ||
                (doc.activeElement instanceof win.HTMLElement ? doc.activeElement : null) ||
                hit.control,
                hit.control,
                hit.wrapper,
                doc.body,
                doc.documentElement,
              ].filter(Boolean);

              const dispatchEscape = (target, type = "keydown") => {{
                target.dispatchEvent(new win.KeyboardEvent(type, {{
                  bubbles: true,
                  cancelable: true,
                  key: "Escape",
                  code: "Escape",
                  keyCode: 27,
                  which: 27,
                }}));
              }};

              keyboardTargets.forEach((target) => {{
                dispatchEscape(target, "keydown");
                dispatchEscape(target, "keyup");
              }});

              win.setTimeout(() => {{
                if (!doc.querySelector('[data-baseweb="popover"] [role="option"], [data-baseweb="menu"] [role="option"]')) {{
                  return;
                }}

                doc.body.dispatchEvent(new win.MouseEvent("mousedown", {{
                  bubbles: true,
                  cancelable: true,
                  clientX: 1,
                  clientY: 1,
                }}));
                doc.body.dispatchEvent(new win.MouseEvent("click", {{
                  bubbles: true,
                  cancelable: true,
                  clientX: 1,
                  clientY: 1,
                }}));
              }}, 0);
            }}

            function stopSelectboxReopen(event, hit) {{
              event.preventDefault();
              event.stopPropagation();
              event.stopImmediatePropagation();
              suppressClickUntil.set(hit.wrapper, Date.now() + 350);
            }}

            function handleSelectboxToggle(event) {{
              const hit = selectboxHit(event);
              if (!hit) {{
                return;
              }}

              if ((suppressClickUntil.get(hit.wrapper) || 0) > Date.now()) {{
                stopSelectboxReopen(event, hit);
                return;
              }}

              if (!selectboxIsExpanded(hit)) {{
                return;
              }}

              stopSelectboxReopen(event, hit);
              sendEscapeToClose(hit);
            }}

            const eventTypes = ["pointerdown", "mousedown", "click"];
            eventTypes.forEach((eventType) => {{
              doc.addEventListener(eventType, handleSelectboxToggle, true);
            }});

            win.__dnascapeSelectboxToggleVersion = selectboxToggleVersion;
            win.__dnascapeSelectboxToggleCleanup = () => {{
              eventTypes.forEach((eventType) => {{
                doc.removeEventListener(eventType, handleSelectboxToggle, true);
              }});
            }};
          }}

          function controlText(node) {{
            return [
              node.getAttribute("aria-label") || "",
              node.getAttribute("title") || "",
              node.getAttribute("data-testid") || "",
              compactText(node),
            ].join(" ").toLowerCase();
          }}

          function isSidebarToggle(button) {{
            const text = controlText(button);
            const compact = text.replace(/\\s+/g, "");
            const namesSidebar = /sidebar/.test(text);
            const namesToggle = /collapse|expand|show|hide|open|close/.test(text);
            const looksLikeChevron =
              /keyboard_double_arrow_(left|right)|chevron_(left|right)|double_arrow|first_page|last_page/.test(compact);
            const livesNearSidebar = Boolean(
              button.closest('section[data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"]')
            );

            return (namesSidebar && namesToggle) || (looksLikeChevron && livesNearSidebar);
          }}

          function sidebarToggleMode(button) {{
            const text = controlText(button);
            const compact = text.replace(/\\s+/g, "");

            if (button.closest('section[data-testid="stSidebar"]')) {{
              return "collapse";
            }}

            if (button.closest('[data-testid="collapsedControl"]')) {{
              return "expand";
            }}

            if (/collapse|hide|close/.test(text) || /keyboard_double_arrow_left|chevron_left|first_page/.test(compact)) {{
              return "collapse";
            }}

            if (/expand|show|open/.test(text) || /keyboard_double_arrow_right|chevron_right|last_page/.test(compact)) {{
              return "expand";
            }}

            return "";
          }}

          function markSidebarToggleButton() {{
            doc
              .querySelectorAll(".dnascape-sidebar-toggle-button, .dnascape-sidebar-toggle-button--collapse, .dnascape-sidebar-toggle-button--expand")
              .forEach((button) => {{
                button.classList.remove(
                  "dnascape-sidebar-toggle-button",
                  "dnascape-sidebar-toggle-button--collapse",
                  "dnascape-sidebar-toggle-button--expand",
                );
              }});

            doc
              .querySelectorAll('button, [role="button"]')
              .forEach((button) => {{
                if (isSidebarToggle(button)) {{
                  button.classList.add("dnascape-sidebar-toggle-button");
                  const mode = sidebarToggleMode(button);
                  if (mode) {{
                    button.classList.add(`dnascape-sidebar-toggle-button--${{mode}}`);
                  }}
                }}
              }});
          }}

          function cssLengthToPx(value, fallback = 12) {{
            const raw = String(value || "").trim();
            const number = parseFloat(raw);
            if (!Number.isFinite(number)) {{
              return fallback;
            }}

            if (raw.endsWith("rem")) {{
              const rootSize = parseFloat(win.getComputedStyle(root).fontSize) || 16;
              return number * rootSize;
            }}

            if (raw.endsWith("em")) {{
              const bodySize = parseFloat(win.getComputedStyle(doc.body).fontSize) || 16;
              return number * bodySize;
            }}

            return number;
          }}

          function syncSidebarFooter() {{
            const footer = doc.querySelector(".sidebar-external-links");
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            const navItems = Array.from(
              doc.querySelectorAll(
                [
                  'section[data-testid="stSidebar"] div[data-testid="stPageLink"]',
                  'section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"]',
                ].join(",")
              )
            ).filter((item) => {{
              const itemBox = item.getBoundingClientRect();
              const itemStyle = win.getComputedStyle(item);
              return (
                itemStyle.display !== "none" &&
                itemStyle.visibility !== "hidden" &&
                itemBox.width > 0 &&
                itemBox.height > 0
              );
            }});

            if (!footer || !sidebar || !navItems.length || sidebar.getAttribute("aria-expanded") === "false") {{
              root.classList.remove("dnascape-sidebar-footer-hidden");
              return;
            }}

            root.classList.remove("dnascape-sidebar-footer-hidden");
            const lastNavItem = navItems.reduce((lowest, item) =>
              item.getBoundingClientRect().bottom > lowest.getBoundingClientRect().bottom
                ? item
                : lowest
            );
            const lastLinkBox = lastNavItem.getBoundingClientRect();
            const footerBox = footer.getBoundingClientRect();
            const minGap = cssLengthToPx(
              win.getComputedStyle(root).getPropertyValue("--dnascape-sidebar-external-min-gap"),
              14,
            );
            const shouldHide = footerBox.top - lastLinkBox.bottom < minGap;
            root.classList.toggle("dnascape-sidebar-footer-hidden", shouldHide);
          }}

          function syncPreviewClearButtons() {{
            doc
              .querySelectorAll(".st-key-plot-preview-clear, .st-key-map-preview-clear, .st-key-simulation-preview-clear")
              .forEach((wrapper) => {{
                wrapper.setAttribute("title", "Clear preview");
                const button = wrapper.querySelector("button");
                if (!button) {{
                  return;
                }}
                button.setAttribute("title", "Clear preview");
                button.setAttribute("aria-label", "Clear preview");
              }});
          }}

          function applyTheme(theme, animate = false) {{
            const selected = normalize(theme);
            const effective = effectiveTheme(selected);

            if (animate) {{
              root.classList.add("dnascape-theme-transition");
              root.getBoundingClientRect();
              win.setTimeout(() => root.classList.remove("dnascape-theme-transition"), 280);
            }}

            root.setAttribute("data-dnascape-theme", selected);
            root.setAttribute("data-dnascape-effective-theme", effective);
            setStoredTheme(selected);
            updateMenuItem(selected);
          }}

          hideBridgeFrame();
          injectStyle();
          removeHeaderExperiment();
          applyTheme(root.getAttribute("data-dnascape-theme") || storedTheme(), false);
          applyHelp(storedHelp());
          bindMenuButton();
          installSelectboxToggleCollapse();
          markSidebarToggleButton();
          syncPreviewClearButtons();
          syncSidebarFooter();
          ensureMenuItem();

          let queued = false;
          const sync = () => {{
            if (queued) {{
              return;
            }}
            queued = true;
            win.requestAnimationFrame(() => {{
              queued = false;
              bindMenuButton();
              markSidebarToggleButton();
              syncPreviewClearButtons();
              syncSidebarFooter();
              ensureMenuItem();
            }});
          }};

          const observer = new MutationObserver(sync);
          observer.observe(doc.body, {{ childList: true, subtree: true }});
          win.addEventListener("resize", sync, {{ passive: true }});

          win.setInterval(() => {{
            bindMenuButton();
            markSidebarToggleButton();
            syncPreviewClearButtons();
            syncSidebarFooter();
            if (menuList()) {{
              ensureMenuItem();
            }}
          }}, 500);

          const systemListener = () => {{
            if (normalize(root.getAttribute("data-dnascape-theme")) === "system") {{
              applyTheme("system", true);
            }}
          }};

          if (prefersDark.addEventListener) {{
            prefersDark.addEventListener("change", systemListener);
          }} else if (prefersDark.addListener) {{
            prefersDark.addListener(systemListener);
          }}
        }})();
        </script>
        """,
        width=1,
        height=1,
        tab_index=-1,
    )


def render_sidebar_logo(
    path: Path,
    alt: str,
    href: str = "/",
    *,
    use_router: bool = False,
) -> None:
    """Render the sidebar PNG without Streamlit's image toolbar."""
    if not path.exists():
        return

    encoded_png = base64.b64encode(path.read_bytes()).decode("ascii")
    escaped_alt = html.escape(alt, quote=True)
    escaped_href = html.escape(href, quote=True)
    logo_markup = f"""
        <div class="dnascape-sidebar-logo">
          <a href="{escaped_href}" target="_self" aria-label="Go to Overview" title="Overview">
            <img src="data:image/png;base64,{encoded_png}" alt="{escaped_alt}" draggable="false">
          </a>
        </div>
    """
    if use_router:
        logo_markup = f"""
            <div class="dnascape-sidebar-logo">
              <img src="data:image/png;base64,{encoded_png}" alt="{escaped_alt}" draggable="false">
            </div>
        """
        with st.container(key="sidebar-logo-home"):
            st.markdown(logo_markup, unsafe_allow_html=True)
            st.button(
                "Overview",
                key="sidebar-logo-home-button",
                on_click=set_active_page,
                args=("overview",),
                width="stretch",
            )
        return

    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )


def render_main_logo(path: Path, alt: str) -> None:
    """Render the main PNG logo without Streamlit's image toolbar."""
    if not path.exists():
        return

    encoded_png = base64.b64encode(path.read_bytes()).decode("ascii")
    escaped_alt = html.escape(alt, quote=True)
    st.markdown(
        f"""
        <div class="dnascape-main-logo">
          <img src="data:image/png;base64,{encoded_png}" alt="{escaped_alt}" draggable="false">
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active_label: str, *, use_router: bool = False) -> None:
    """Render the shared left navigation."""
    active_slug = normalize_page(active_label)
    active_url = next((url for slug, _, url in NAV_ITEMS if slug == active_slug), "")
    active_route = active_url.strip("/")
    with st.sidebar:
        page_link_active_selector = """
                section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[aria-current="page"]) a,"""
        page_link_active_hover_selector = """
                section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[aria-current="page"]) a:hover,
                section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[aria-current="page"]) a:focus-visible,"""
        page_link_active_child_selector = """
            section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[aria-current="page"]) a *"""
        if active_route:
            page_link_active_selector += f"""
                section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[href*="{active_route}"]) a,"""
            page_link_active_hover_selector += f"""
                section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[href*="{active_route}"]) a:hover,
                section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[href*="{active_route}"]) a:focus-visible,"""
            page_link_active_child_selector += f""",
            section[data-testid="stSidebar"] div[data-testid="stPageLink"]:has(a[href*="{active_route}"]) a *"""

        st.markdown(
            f"""
            <span class="dnascape-sidebar-active-style" aria-hidden="true"></span>
            <style>
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[kind="secondary"]:not(:hover):not(:focus-visible),
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"]:not([class*="st-key-sidebar-nav-{active_slug}"]) button:not([kind="primary"]):not(:hover):not(:focus-visible) {{
              background: transparent !important;
              border-color: transparent !important;
              box-shadow: none !important;
            }}

            {page_link_active_selector}
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[kind="primary"],
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-{active_slug}"] button {{
              background: var(--dnascape-sidebar-active-bg) !important;
              color: var(--dnascape-text) !important;
              font-weight: 500 !important;
              justify-content: flex-start !important;
              text-align: left !important;
            }}

            {page_link_active_hover_selector}
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[kind="primary"]:hover,
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[kind="primary"]:focus-visible,
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-{active_slug}"] button:hover,
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-{active_slug}"] button:focus-visible {{
              background: var(--dnascape-hover-bg) !important;
              color: var(--dnascape-text) !important;
              outline: none !important;
              text-decoration: none !important;
            }}

            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-"] button[kind="primary"] *,
            section[data-testid="stSidebar"] [class*="st-key-sidebar-nav-{active_slug}"] button *,
            {page_link_active_child_selector} {{
              color: var(--dnascape-text) !important;
              text-align: left !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        render_sidebar_logo(
            SIDEBAR_LOGO_PATH,
            "DNAscape",
            "?page=overview" if use_router else "/",
            use_router=use_router,
        )

        if use_router:
            for slug, label, _ in NAV_ITEMS:
                st.button(
                    label,
                    key=f"sidebar-nav-{slug}",
                    on_click=set_active_page,
                    args=(slug,),
                    type="primary" if slug == active_slug else "secondary",
                    width="stretch",
                )
        else:
            render_page_link("streamlit_app.py", "Overview", "/")
            render_page_link("pages/1_Data_Management.py", "Data Management", "/Data_Management")
            render_page_link("pages/2_Mapping.py", "Mapping", "/Mapping")
            render_page_link("pages/3_Simulation.py", "Simulation", "/Simulation")
            render_page_link("pages/4_Plotting.py", "Plotting", "/Plotting")
            render_page_link("pages/5_Cached_Files.py", "Cached Files", "/Cached_Files")
            render_page_link("pages/6_Resources.py", "Resources", "/Resources")

        st.markdown(
            f"""
            <div class="sidebar-external-links">
              <div class="sidebar-external-icons">
                <a href="{GITHUB_URL}" aria-label="GitHub" title="GitHub">
                  <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                    <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.61 7.61 0 0 1 8 3.86c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
                  </svg>
                </a>
                <a href="{DOCS_URL}" aria-label="Documentation" title="Documentation">
                  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M12 6.15c-1.5-1.03-3.25-1.55-5.25-1.55H4.8c-.44 0-.8.36-.8.8v12.65c0 .44.36.8.8.8h1.95c2 0 3.75.52 5.25 1.55V6.15Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/>
                    <path d="M12 6.15c1.5-1.03 3.25-1.55 5.25-1.55h1.95c.44 0 .8.36.8.8v12.65c0 .44-.36.8-.8.8h-1.95c-2 0-3.75.52-5.25 1.55V6.15Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/>
                    <path d="M12 6.15v14.25" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
                  </svg>
                </a>
                <a href="{GITHUB_URL}/issues" aria-label="Report an issue or suggest a feature" title="Report an issue or suggest a feature">
                  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M9 18h6" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
                    <path d="M10 22h4" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
                    <path d="M8.75 14.75c-1.55-1.05-2.5-2.8-2.5-4.72A5.75 5.75 0 0 1 12 4.28a5.75 5.75 0 0 1 5.75 5.75c0 1.92-.95 3.67-2.5 4.72-.68.46-1.08 1.22-1.08 2.04v.21H9.83v-.21c0-.82-.4-1.58-1.08-2.04Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/>
                    <path d="M12 1.75v1M4.8 3.65l.7.7M19.2 3.65l-.7.7M2.25 10.25h1M20.75 10.25h1" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.6"/>
                  </svg>
                </a>
              </div>
              <div class="sidebar-app-version">v{html.escape(APP_VERSION)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_link(page: str, label: str, url: str) -> None:
    """Render an internal Streamlit page link with a simple fallback."""
    try:
        st.page_link(page, label=label)
    except Exception:
        st.markdown(f"<a class='fallback-link' href='{url}'>{label}</a>", unsafe_allow_html=True)


def render_page_intro(kicker: str, title: str, body: str) -> None:
    """Render a consistent page heading."""
    st.markdown(
        f"""
        <section class="hero">
          <p class="eyebrow">{kicker}</p>
          <h1>{title}</h1>
          <p class="lead">{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
