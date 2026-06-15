"""Shared layout helpers for the DNAscape Streamlit prototype."""

import base64
import html
import json
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
LOGO_PATH = ROOT_DIR / "docs" / "source" / "_static" / "dnascape_logo.png"
SIDEBAR_LOGO_PATH = ROOT_DIR / "docs" / "source" / "_static" / "dnascape_logo_small.png"
TAB_LOGO_PATH = ROOT_DIR / "docs" / "source" / "_static" / "dnascape_logo_tab.png"
GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"
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
          --dnascape-card-hover-bg: color-mix(in srgb, var(--dnascape-accent) 4%, var(--dnascape-panel));
          --dnascape-shadow: rgba(16, 24, 40, 0.04);
          --dnascape-shadow-hover: rgba(16, 24, 40, 0.07);
          --dnascape-sidebar-width: clamp(18rem, 20vw, 20rem);
          --dnascape-sidebar-inner-width: min(82%, 16rem);
          --dnascape-sidebar-logo-width: min(58%, 10.25rem);
          --dnascape-sidebar-nav-font-size: 1.06rem;
          --dnascape-sidebar-nav-gap: 0.58rem;
          --dnascape-sidebar-nav-padding-y: 0.44rem;
          --dnascape-sidebar-nav-padding-x: 0.85rem;
          --dnascape-sidebar-external-gap: 1.8rem;
          --dnascape-sidebar-toggle-inset: 0.9rem;
        }

        :root[data-dnascape-effective-theme="light"] {
          color-scheme: light;
        }

        :root[data-dnascape-effective-theme="dark"] {
          --background-color: #151413;
          --secondary-background-color: #211f1d;
          --text-color: #f3eee8;
          --dnascape-accent: color-mix(in srgb, var(--primary-color, #b03412) 58%, #ffffff);
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
            circle at 48.5% 2%,
            transparent 0 2.4%,
            #000 2.8%
          );
          -webkit-mask-repeat: no-repeat;
          -webkit-mask-size: 100% 100%;
          mask-image: radial-gradient(
            circle at 48.5% 2%,
            transparent 0 2.4%,
            #000 2.8%
          );
          mask-repeat: no-repeat;
          mask-size: 100% 100%;
        }

        :root.dnascape-theme-transition,
        :root.dnascape-theme-transition * {
          transition: background-color 160ms ease, border-color 160ms ease,
            box-shadow 160ms ease, color 160ms ease, filter 160ms ease !important;
        }

        [data-testid="stAppViewContainer"] {
          background: var(--dnascape-bg);
          color: var(--dnascape-text);
        }

        header[data-testid="stHeader"] {
          background: color-mix(in srgb, var(--dnascape-bg) 94%, transparent);
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

        section[data-testid="stSidebar"],
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"] {
          background: var(--dnascape-sidebar-bg);
        }

        section[data-testid="stSidebar"] {
          box-sizing: border-box;
          border-right: 0 !important;
          color: var(--dnascape-text);
          position: relative;
        }

        section[data-testid="stSidebar"] button[aria-label*="sidebar" i],
        section[data-testid="stSidebar"] button[title*="sidebar" i],
        section[data-testid="stSidebar"] button[data-testid^="stBaseButton-header"] {
          margin-right: var(--dnascape-sidebar-toggle-inset) !important;
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
          padding-left: 0 !important;
          padding-right: 0 !important;
          scrollbar-gutter: auto !important;
        }

        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"] {
          box-sizing: border-box !important;
          max-width: 100% !important;
          min-width: 0 !important;
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

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
          align-items: center;
          gap: 0.18rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] {
          margin-left: auto !important;
          margin-right: auto !important;
          margin-bottom: var(--dnascape-sidebar-nav-gap);
          width: var(--dnascape-sidebar-inner-width) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a {
          box-sizing: border-box;
          color: var(--dnascape-text) !important;
          display: flex;
          font-size: var(--dnascape-sidebar-nav-font-size);
          font-weight: 500;
          justify-content: flex-start;
          line-height: 1.35;
          padding: var(--dnascape-sidebar-nav-padding-y)
            var(--dnascape-sidebar-nav-padding-x);
          text-decoration: none !important;
          width: 100%;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a *,
        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"],
        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a[aria-current="page"] * {
          color: var(--dnascape-text) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover {
          background: var(--dnascape-hover-bg);
          color: var(--dnascape-text) !important;
          text-decoration: none !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stPageLink"] a:hover * {
          color: var(--dnascape-text) !important;
        }

        .dnascape-sidebar-logo {
          display: block;
          line-height: 0;
          margin: 0 auto 1.42rem;
          width: var(--dnascape-sidebar-logo-width) !important;
        }

        .dnascape-sidebar-logo img {
          display: block;
          height: auto;
          pointer-events: none;
          user-select: none;
          width: 100%;
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
          max-width: 1500px;
          padding-top: 2.25rem;
          padding-bottom: 4rem;
        }

        h1, h2, h3 {
          letter-spacing: 0;
          color: var(--dnascape-text);
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
          margin: 1.5rem 0 1.8rem;
          max-width: 780px;
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
          display: block;
          min-height: 172px;
          padding: 1.2rem 1.15rem;
          text-decoration: none !important;
          transition: background-color 140ms ease, border-color 140ms ease,
            box-shadow 140ms ease;
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

        .workflow-card::after {
          content: none !important;
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
          margin-left: auto;
          margin-right: auto;
          margin-top: var(--dnascape-sidebar-external-gap);
          width: var(--dnascape-sidebar-inner-width);
        }

        .sidebar-external-links a {
          color: var(--dnascape-accent) !important;
          display: block;
          font-size: 0.95rem;
          line-height: 1.45;
          padding: 0.24rem 0;
          text-decoration: none;
        }

        .sidebar-external-links a:hover {
          color: var(--dnascape-accent) !important;
          text-decoration: underline;
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
            padding-top: 1.3rem;
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


def _render_header_theme_toggle_experiment() -> None:
    """Previous header-button experiment, kept out of the active app shell."""
    st.iframe(
        f"""
        <script>
        (() => {{
          const parentWindow = window.parent;
          const doc = parentWindow.document;
          const root = doc.documentElement;
          const storageKey = "dnascape-theme";
          const toggleId = "dnascape-theme-toggle";
          const styleId = "dnascape-theme-toggle-style";
          const toggleVersion = "parent-header-button-1";
          const themes = {list(THEMES)!r};
          const icons = {json.dumps(THEME_ICONS)};
          const labels = {THEME_LABELS!r};
          const prefersDark = parentWindow.matchMedia("(prefers-color-scheme: dark)");

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
              return normalize(parentWindow.localStorage.getItem(storageKey));
            }} catch (error) {{
              return "light";
            }}
          }}

          function setStoredTheme(theme) {{
            try {{
              parentWindow.localStorage.setItem(storageKey, theme);
            }} catch (error) {{
              /* Storage can be unavailable in hardened browser contexts. */
            }}
          }}

          function injectStyle() {{
            if (doc.getElementById(styleId)) {{
              return;
            }}

            const style = doc.createElement("style");
            style.id = styleId;
            style.textContent = `
              #${{toggleId}} {{
                align-items: center;
                background: transparent;
                border: 0;
                border-radius: 8px;
                color: inherit;
                cursor: pointer;
                display: inline-flex;
                flex: 0 0 auto;
                font: inherit;
                height: 2.25rem;
                justify-content: center;
                line-height: 1;
                margin-left: 0.125rem;
                margin-right: 0.125rem;
                min-height: 2.25rem;
                min-width: 2.25rem;
                padding: 0;
                user-select: none;
                width: 2.25rem;
              }}

              #${{toggleId}}:hover,
              #${{toggleId}}:focus-visible {{
                background: color-mix(in srgb, currentColor 10%, transparent);
                outline: none;
              }}

              #${{toggleId}}:active {{
                background: color-mix(in srgb, currentColor 16%, transparent);
              }}

              #${{toggleId}}.dnascape-theme-toggle--fallback {{
                position: fixed;
                right: 2.75rem;
                top: calc((3rem - 2.25rem) / 2);
                z-index: 2147483647;
              }}

              #${{toggleId}} .dnascape-theme-icon-slot {{
                align-items: center;
                display: inline-flex;
                justify-content: center;
                line-height: 0;
              }}

              #${{toggleId}} .dnascape-theme-icon {{
                display: block;
                height: 1.15rem;
                line-height: 1;
                width: 1.15rem;
              }}
            `;
            doc.head.appendChild(style);
          }}

          function mainMenuNode() {{
            const selectors = [
              '[data-testid="stMainMenu"]',
              '.stMainMenu',
              '#MainMenu',
            ];

            for (const selector of selectors) {{
              const menu = doc.querySelector(selector);
              if (menu && menu.parentElement) {{
                return menu;
              }}
            }}

            return null;
          }}

          function headerTarget() {{
            const menu = mainMenuNode();
            if (!menu || !menu.parentElement) {{
              return null;
            }}
            return {{ parent: menu.parentElement, before: menu }};
          }}

          function resetButtonAttributes(toggle) {{
            [
              "aria-controls",
              "aria-expanded",
              "aria-haspopup",
              "aria-pressed",
              "data-clicked",
            ].forEach((attribute) => toggle.removeAttribute(attribute));

            toggle.id = toggleId;
            toggle.type = "button";
            toggle.setAttribute("kind", "headerNoPadding");
            toggle.setAttribute("data-testid", "dnascapeThemeToggle");
            toggle.dataset.dnascapeVersion = toggleVersion;
          }}

          function createIconSlot() {{
            const icon = document.createElement("span");
            icon.className = "dnascape-theme-icon-slot";
            icon.setAttribute("aria-hidden", "true");
            return icon;
          }}

          function createToggle() {{
            const toggle = doc.createElement("button");
            toggle.className = "dnascape-theme-toggle";
            toggle.replaceChildren(createIconSlot());
            resetButtonAttributes(toggle);
            return toggle;
          }}

          function removeStaleToggles() {{
            doc
              .querySelectorAll(`#${{toggleId}}, .dnascape-theme-toggle, [data-testid="dnascapeThemeToggle"]`)
              .forEach((node) => {{
                if (node.dataset.dnascapeVersion !== toggleVersion) {{
                  node.remove();
                }}
              }});
          }}

          function ensureToggle() {{
            injectStyle();
            removeStaleToggles();

            const menuTarget = headerTarget();
            let toggle = doc.getElementById(toggleId);

            if (!toggle) {{
              toggle = createToggle();
            }}

            if (!toggle) {{
              return null;
            }}

            doc.querySelectorAll(`#${{toggleId}}`).forEach((node) => {{
              if (node !== toggle) {{
                node.remove();
              }}
            }});

            if (menuTarget && menuTarget.parent) {{
              toggle.className = "dnascape-theme-toggle";
              toggle.classList.remove("dnascape-theme-toggle--fallback");
              resetButtonAttributes(toggle);

              if (toggle.parentElement !== menuTarget.parent || toggle.nextSibling !== menuTarget.before) {{
                menuTarget.parent.insertBefore(toggle, menuTarget.before);
              }}
            }} else {{
              toggle.className = "dnascape-theme-toggle dnascape-theme-toggle--fallback";
              resetButtonAttributes(toggle);
              doc.body.appendChild(toggle);
            }}

            return toggle;
          }}

          function updateToggle(theme) {{
            ensureToggle();
            doc.querySelectorAll(".dnascape-theme-toggle").forEach((toggle) => {{
              const icon = toggle.querySelector(".dnascape-theme-icon-slot");
              if (icon && icon.dataset.theme !== theme) {{
                icon.innerHTML = icons[theme];
                icon.dataset.theme = theme;
              }}
              const label = `Theme: ${{labels[theme]}}.`;
              toggle.setAttribute("aria-label", label);
              toggle.setAttribute("title", label);
            }});
          }}

          function applyTheme(theme, animate = false) {{
            const selected = normalize(theme);
            const effective = effectiveTheme(selected);

            if (animate) {{
              root.classList.add("dnascape-theme-transition");
              window.setTimeout(() => root.classList.remove("dnascape-theme-transition"), 190);
            }}

            root.setAttribute("data-dnascape-theme", selected);
            root.setAttribute("data-dnascape-effective-theme", effective);
            setStoredTheme(selected);
            updateToggle(selected);
          }}

          function hideBuiltInThemeMenuItems() {{
            const patterns = [/^settings$/i, /^theme$/i, /choose app theme/i, /appearance/i];
            doc.querySelectorAll('[role="menuitem"], [role="option"], [role="menuitemradio"], button, a').forEach((node) => {{
              const text = (node.textContent || "").replace(/\\s+/g, " ").trim();
              if (!patterns.some((pattern) => pattern.test(text))) {{
                return;
              }}
              const menuItem = node.closest('[role="menuitem"], [role="option"], [role="menuitemradio"], li, div') || node;
              if (menuItem.classList && menuItem.classList.contains("dnascape-theme-toggle")) {{
                return;
              }}
              menuItem.style.display = "none";
            }});
          }}

          function bindToggle() {{
            const toggle = ensureToggle();
            if (!toggle || toggle.dataset.dnascapeBound === "true") {{
              return;
            }}
            toggle.dataset.dnascapeBound = "true";
            toggle.addEventListener("click", (event) => {{
              event.preventDefault();
              const current = normalize(root.getAttribute("data-dnascape-theme") || storedTheme());
              applyTheme(nextTheme(current), true);
            }});
          }}

          bindToggle();
          applyTheme(root.getAttribute("data-dnascape-theme") || storedTheme(), false);
          hideBuiltInThemeMenuItems();

          let syncQueued = false;
          function syncHeaderButton() {{
            if (syncQueued) {{
              return;
            }}
            syncQueued = true;
            parentWindow.requestAnimationFrame(() => {{
              syncQueued = false;
              bindToggle();
              updateToggle(normalize(root.getAttribute("data-dnascape-theme") || storedTheme()));
              hideBuiltInThemeMenuItems();
            }});
          }}

          const observer = new MutationObserver(() => {{
            syncHeaderButton();
          }});
          observer.observe(doc.body, {{ childList: true, subtree: true }});

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


def _render_theme_toggle_menu_experiment() -> None:
    """Previous direct st.html menu experiment, kept out of the active shell."""
    st.html(
        f"""
        <script>
        (() => {{
          const root = document.documentElement;
          const storageKey = "dnascape-theme";
          const styleId = "dnascape-theme-menu-style";
          const styleVersion = "theme-menu-critical-paint-4";
          const wrapperId = "dnascape-theme-menu-item-wrapper";
          const itemId = "dnascape-theme-menu-item";
          const themes = {list(THEMES)!r};
          const icons = {json.dumps(THEME_ICONS)};
          const labels = {THEME_LABELS!r};
          const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

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
              return normalize(window.localStorage.getItem(storageKey));
            }} catch (error) {{
              return "light";
            }}
          }}

          function setStoredTheme(theme) {{
            try {{
              window.localStorage.setItem(storageKey, theme);
            }} catch (error) {{
              /* Storage can be unavailable in hardened browser contexts. */
            }}
          }}

          function injectStyle() {{
            if (document.getElementById(styleId)) {{
              return;
            }}

            const style = document.createElement("style");
            style.id = styleId;
            style.textContent = `
              #${{wrapperId}} {{
                list-style: none;
                margin: 0;
                padding: 0;
              }}

              #${{itemId}} {{
                align-items: center;
                background: transparent;
                border: 0;
                color: inherit;
                cursor: pointer;
                display: flex;
                font: inherit;
                gap: 0.65rem;
                justify-content: flex-start;
                line-height: 1.35;
                min-height: 2.25rem;
                padding: 0.48rem 0.75rem;
                text-align: left;
                width: 100%;
              }}

              #${{itemId}}:hover,
              #${{itemId}}:focus-visible {{
                background: color-mix(in srgb, currentColor 10%, transparent);
                outline: none;
              }}

              #${{itemId}} .dnascape-theme-menu-icon {{
                align-items: center;
                display: inline-flex;
                flex: 0 0 1.1rem;
                justify-content: center;
                line-height: 0;
              }}

              #${{itemId}} .dnascape-theme-menu-icon svg {{
                display: block;
                height: 1.08rem;
                width: 1.08rem;
              }}

              #${{itemId}} .dnascape-theme-menu-label {{
                white-space: nowrap;
              }}
            `;
            document.head.appendChild(style);
          }}

          function removeHeaderExperiment() {{
            document
              .querySelectorAll("#dnascape-theme-toggle, .dnascape-theme-toggle, [data-testid='dnascapeThemeToggle']")
              .forEach((node) => node.remove());

            const oldStyle = document.getElementById("dnascape-theme-toggle-style");
            if (oldStyle) {{
              oldStyle.remove();
            }}
          }}

          function menuList() {{
            const popover = document.querySelector(
              '[data-testid="stMainMenuPopover"], .stMainMenuPopover'
            );
            if (!popover) {{
              return null;
            }}
            return (
              popover.querySelector('[data-testid="stMainMenuList"], [role="menu"]') ||
              popover
            );
          }}

          function updateMenuItem(theme) {{
            const item = document.getElementById(itemId);
            if (!item) {{
              return;
            }}

            const label = `Theme: ${{labels[theme]}}`;
            item.setAttribute("aria-label", `${{label}}.`);
            item.setAttribute("title", `${{label}}.`);
            item.querySelector(".dnascape-theme-menu-icon").innerHTML = icons[theme];
            item.querySelector(".dnascape-theme-menu-label").textContent = label;
          }}

          function createMenuItem() {{
            const wrapper = document.createElement("div");
            wrapper.id = wrapperId;
            wrapper.setAttribute("role", "none");

            const item = document.createElement("button");
            item.id = itemId;
            item.type = "button";
            item.setAttribute("role", "menuitem");
            item.innerHTML = `
              <span class="dnascape-theme-menu-icon" aria-hidden="true"></span>
              <span class="dnascape-theme-menu-label"></span>
            `;

            item.addEventListener("mousedown", (event) => {{
              event.stopPropagation();
            }});

            item.addEventListener("click", (event) => {{
              event.preventDefault();
              event.stopPropagation();
              const current = normalize(root.getAttribute("data-dnascape-theme") || storedTheme());
              applyTheme(nextTheme(current), true);
            }});

            wrapper.appendChild(item);
            return wrapper;
          }}

          function ensureMenuItem() {{
            injectStyle();
            const list = menuList();
            if (!list) {{
              return;
            }}

            let wrapper = document.getElementById(wrapperId);
            if (!wrapper) {{
              wrapper = createMenuItem();
            }}

            if (wrapper.parentElement !== list) {{
              list.insertBefore(wrapper, list.firstElementChild);
            }} else if (list.firstElementChild !== wrapper) {{
              list.insertBefore(wrapper, list.firstElementChild);
            }}

            updateMenuItem(normalize(root.getAttribute("data-dnascape-theme") || storedTheme()));
          }}

          function applyTheme(theme, animate = false) {{
            const selected = normalize(theme);
            const effective = effectiveTheme(selected);

            if (animate) {{
              root.classList.add("dnascape-theme-transition");
              window.setTimeout(() => root.classList.remove("dnascape-theme-transition"), 190);
            }}

            root.setAttribute("data-dnascape-theme", selected);
            root.setAttribute("data-dnascape-effective-theme", effective);
            setStoredTheme(selected);
            updateMenuItem(selected);
          }}

          removeHeaderExperiment();
          applyTheme(root.getAttribute("data-dnascape-theme") || storedTheme(), false);
          ensureMenuItem();

          let syncQueued = false;
          const syncMenu = () => {{
            if (syncQueued) {{
              return;
            }}
            syncQueued = true;
            window.requestAnimationFrame(() => {{
              syncQueued = false;
              removeHeaderExperiment();
              ensureMenuItem();
            }});
          }};

          const observer = new MutationObserver(syncMenu);
          observer.observe(document.body, {{ childList: true, subtree: true }});

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
        unsafe_allow_javascript=True,
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
          const styleId = "dnascape-theme-menu-style";
          const styleVersion = "theme-menu-critical-paint-9";
          const wrapperId = "dnascape-theme-menu-item-wrapper";
          const itemId = "dnascape-theme-menu-item";
          const themes = {list(THEMES)!r};
          const icons = {json.dumps(THEME_ICONS)};
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

          function injectStyle() {{
            const existing = doc.getElementById(styleId);
            if (existing && existing.dataset.dnascapeVersion === styleVersion) {{
              return;
            }}

            const style = existing || doc.createElement("style");
            style.id = styleId;
            style.dataset.dnascapeVersion = styleVersion;
            style.textContent = `
              :root[data-dnascape-effective-theme="dark"] {{
                --background-color: #151413;
                --secondary-background-color: #211f1d;
                --text-color: #f3eee8;
                --dnascape-bg: #151413;
                --dnascape-panel: #211f1d;
                --dnascape-text: #f3eee8;
                --dnascape-sidebar-bg: #1b1918;
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

              #${{wrapperId}} {{
                display: block;
                list-style: none;
                margin: 0;
                padding: 0;
              }}

              #${{itemId}} {{
                align-items: center;
                box-sizing: border-box;
                cursor: pointer;
                display: flex;
                gap: 0.3rem;
                justify-content: flex-start;
                width: 100%;
              }}

              #${{itemId}}:not(.dnascape-native-menu-item) {{
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
              #${{itemId}}:hover,
              #${{itemId}}:focus-visible {{
                background: rgba(250, 250, 250, 0.10) !important;
                border-radius: 0.5rem !important;
                outline: none !important;
              }}

              :root[data-dnascape-effective-theme="dark"] #${{wrapperId}}:hover #${{itemId}},
              :root[data-dnascape-effective-theme="dark"] #${{wrapperId}}:focus-within #${{itemId}},
              :root[data-dnascape-effective-theme="dark"] #${{itemId}}:hover,
              :root[data-dnascape-effective-theme="dark"] #${{itemId}}:focus-visible {{
                background: rgba(250, 250, 250, 0.10) !important;
              }}

              #${{itemId}} .dnascape-theme-menu-icon {{
                align-items: center;
                color: currentColor;
                display: inline-flex;
                flex: 0 0 auto;
                justify-content: center;
                line-height: 0;
                pointer-events: none;
              }}

              #${{itemId}} .dnascape-theme-menu-icon svg {{
                display: block;
                height: 1.08rem;
                pointer-events: none;
                width: 1.08rem;
              }}

              #${{itemId}} .dnascape-theme-menu-icon * {{
                pointer-events: none;
              }}

              #${{itemId}} .dnascape-theme-menu-label {{
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
              .filter((node) => node.id !== itemId && !node.closest(`#${{wrapperId}}`));

            return (
              candidates.find((node) => /about|report a bug|get help/i.test(node.textContent || "")) ||
              candidates[0] ||
              null
            );
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

          function stopMenuPointer(event) {{
            event.stopPropagation();
          }}

          function activateMenuItem(event) {{
            event.preventDefault();
            event.stopImmediatePropagation();
            const current = normalize(root.getAttribute("data-dnascape-theme") || storedTheme());
            applyTheme(nextTheme(current), true);
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

            bindMenuItem(wrapper.querySelector(`#${{itemId}}`));

            if (wrapper.parentElement !== list) {{
              list.insertBefore(wrapper, list.firstElementChild);
            }} else if (list.firstElementChild !== wrapper) {{
              list.insertBefore(wrapper, list.firstElementChild);
            }}

            syncNativeMenuStyle(list, wrapper, wrapper.querySelector(`#${{itemId}}`));
            updateMenuItem(normalize(root.getAttribute("data-dnascape-theme") || storedTheme()));
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

          function applyTheme(theme, animate = false) {{
            const selected = normalize(theme);
            const effective = effectiveTheme(selected);

            if (animate) {{
              root.classList.add("dnascape-theme-transition");
              win.setTimeout(() => root.classList.remove("dnascape-theme-transition"), 190);
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
          bindMenuButton();
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
              ensureMenuItem();
            }});
          }};

          const observer = new MutationObserver(sync);
          observer.observe(doc.body, {{ childList: true, subtree: true }});

          win.setInterval(() => {{
            bindMenuButton();
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


def render_sidebar_logo(path: Path, alt: str) -> None:
    """Render the sidebar PNG without Streamlit's image toolbar."""
    if not path.exists():
        return

    encoded_png = base64.b64encode(path.read_bytes()).decode("ascii")
    escaped_alt = html.escape(alt, quote=True)
    st.markdown(
        f"""
        <div class="dnascape-sidebar-logo">
          <img src="data:image/png;base64,{encoded_png}" alt="{escaped_alt}" draggable="false">
        </div>
        """,
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


def render_sidebar(active_label: str) -> None:
    """Render the shared left navigation."""
    with st.sidebar:
        render_sidebar_logo(SIDEBAR_LOGO_PATH, "DNAscape")

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
              <a href="{GITHUB_URL}">GitHub</a>
              <a href="{DOCS_URL}">Documentation</a>
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
