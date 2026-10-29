"""Sphinx configuration for the app-first DNAscape user guide."""

from datetime import datetime

project = "DNAscape"
author = "Francisco Berkemeier"
copyright = f"{datetime.now().year}, {author}"
release = "1.0.0"

extensions = ["sphinx.ext.mathjax", "sphinx_copybutton"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_title = "DNAscape documentation"
html_short_title = "DNAscape"
html_favicon = "_static/dnascape_logo_tab.png"
html_static_path = ["_static"]
html_css_files = ["dnascape_theme.css"]
html_show_sourcelink = False
html_use_index = False
html_theme_options = {
    "version_selector": False,
    "language_selector": False,
    "flyout_display": "hidden",
    "collapse_navigation": True,
    "navigation_depth": 2,
}
