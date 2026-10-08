"""Compatibility imports for :mod:`dnascape.workflows.exports`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import exports as _workflow
from dnascape.workflows.exports import (
    AxisKind as AxisKind,
    EXPORT_FORMATS as EXPORT_FORMATS,
    ExportArtifact as ExportArtifact,
    MAX_BINARY_EXPORT_BYTES as MAX_BINARY_EXPORT_BYTES,
    MAX_CHART_CAPTURE_BYTES as MAX_CHART_CAPTURE_BYTES,
    MAX_PLOT_PIXELS as MAX_PLOT_PIXELS,
    MAX_PLOT_SERIES_POINTS as MAX_PLOT_SERIES_POINTS,
    MAX_PLOT_XT_COLUMNS as MAX_PLOT_XT_COLUMNS,
    MAX_PLOT_XT_ROWS as MAX_PLOT_XT_ROWS,
    MAX_TEXT_EXPORT_VALUES as MAX_TEXT_EXPORT_VALUES,
    PLOT_FORMATS as PLOT_FORMATS,
    RAW_FORMATS as RAW_FORMATS,
    SelectedOutput as SelectedOutput,
    build_chart_capture_export as build_chart_capture_export,
    build_plot_export as build_plot_export,
    build_raw_export as build_raw_export,
    compressed_export_artifact as compressed_export_artifact,
    normalize_export_format as normalize_export_format,
    safe_export_filename as safe_export_filename,
    select_result_output as select_result_output,
)

sys.modules[__name__] = _workflow
