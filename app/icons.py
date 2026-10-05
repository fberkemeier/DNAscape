"""Code-native symbols shared across the DNAscape interface."""

from __future__ import annotations

DATA_ICON = "dnascape_data"

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

__all__ = ["DATA_ICON", "DATA_ICON_SVG", "OUTLINE_ICON_SVGS", "PLOT_ICON_SVG"]
