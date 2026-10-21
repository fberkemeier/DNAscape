"""Code-native symbols shared across the DNAscape interface."""

from __future__ import annotations

DATA_ICON = "dnascape_data"

# Quasar renders SVG paths inline, so this counterclockwise arrow inherits
# each button's colour in both themes instead of using a fixed-colour image.
RESET_ICON = (
    "M4 12a9 9 0 1 1 2.64 6.36l1.77-1.77A6.5 6.5 0 1 0 6.5 12Z "
    "M1 9h8.5L5.25 13.25Z"
)

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

OUTLINE_ICON_SVGS = {
    "overview": """
<svg class="dn-outline-icon" viewBox="0 0 24 24" width="1em" height="1em"
     aria-hidden="true" focusable="false">
  <path d="m3.5 11.25 8.5-7.5 8.5 7.5"/>
  <path d="M5.75 9.25v10.5h12.5V9.25M9.5 19.75v-6.5h5v6.5"/>
</svg>
""",
    "data": """
<svg class="dn-outline-icon" viewBox="0 0 24 24" width="1em" height="1em"
     aria-hidden="true" focusable="false">
  <ellipse cx="12" cy="5.25" rx="7.5" ry="3"
           fill="none" stroke="currentColor"/>
  <path d="M4.5 5.25v5c0 1.65 3.36 3 7.5 3s7.5-1.35 7.5-3v-5"/>
  <path d="M4.5 10.25v5c0 1.65 3.36 3 7.5 3s7.5-1.35 7.5-3v-5"/>
</svg>
""",
    "mapping": """
<svg class="dn-outline-icon" viewBox="0 0 24 24" width="1em" height="1em"
     aria-hidden="true" focusable="false">
  <path d="m3.5 5.5 5.25-2.25 6.5 2.25 5.25-2.25v15.25l-5.25 2.25-6.5-2.25-5.25 2.25z"/>
  <path d="M8.75 3.25V18.5M15.25 5.5v15.25"/>
</svg>
""",
    "simulation": """
<svg class="dn-outline-icon" viewBox="0 0 24 24" width="1em" height="1em"
     aria-hidden="true" focusable="false">
  <rect x="3" y="3.75" width="18" height="13.5" rx="2"/>
  <path d="M8 21h8M12 17.25V21"/>
  <path d="M6.25 11.5h2.1l1.55-3.25 3.2 6.25 1.55-3h3.1"/>
</svg>
""",
    "cached": """
<svg class="dn-outline-icon" viewBox="0 0 24 24" width="1em" height="1em"
     aria-hidden="true" focusable="false">
  <rect x="4" y="7.25" width="16" height="13" rx="1.5"/>
  <rect x="3" y="3.25" width="18" height="4" rx="1"/>
  <path d="M9.25 11.5h5.5"/>
</svg>
""",
    "resources": """
<svg class="dn-outline-icon" viewBox="0 0 24 24" width="1em" height="1em"
     aria-hidden="true" focusable="false">
  <path d="M12 6.5c-1.8-1.55-4.35-2.25-7.5-2.1v14.75c3.15-.15 5.7.55 7.5 2.1"/>
  <path d="M12 6.5c1.8-1.55 4.35-2.25 7.5-2.1v14.75c-3.15-.15-5.7.55-7.5 2.1M12 6.5v14.75"/>
</svg>
""",
}

DATA_ICON_SVG = OUTLINE_ICON_SVGS["data"]

PLOT_ICON_SVG = """
<svg class="dn-plot-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path class="dn-plot-icon-axis" d="M4 3.5V20h16.5" />
  <path class="dn-plot-icon-line" d="m7 16 4-5 3.2 3.1L19 7.5" />
  <circle class="dn-plot-icon-point" cx="7" cy="16" r="1" />
  <circle class="dn-plot-icon-point" cx="11" cy="11" r="1" />
  <circle class="dn-plot-icon-point" cx="14.2" cy="14.1" r="1" />
  <circle class="dn-plot-icon-point" cx="19" cy="7.5" r="1" />
</svg>
"""

__all__ = ["DATA_ICON", "DATA_ICON_SVG", "GITHUB_ICON_SVG", "OUTLINE_ICON_SVGS", "PLOT_ICON_SVG", "RESET_ICON"]
