"""Compatibility imports for :mod:`dnascape.workflows.result_naming`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import result_naming as _workflow
from dnascape.workflows.result_naming import (
    NAME_STYLE_OPTIONS as NAME_STYLE_OPTIONS,
    NameStyle as NameStyle,
    OUTPUT_FILE_TOKENS as OUTPUT_FILE_TOKENS,
    PLOTTING_FILE_PREFIX as PLOTTING_FILE_PREFIX,
    RDATA_FILE_PREFIX as RDATA_FILE_PREFIX,
    RMAP_FILE_PREFIX as RMAP_FILE_PREFIX,
    RSIM_FILE_PREFIX as RSIM_FILE_PREFIX,
    ResultName as ResultName,
    combination_result_names as combination_result_names,
    compact_component as compact_component,
    compact_export_stem as compact_export_stem,
    normalize_name_style as normalize_name_style,
    output_file_token as output_file_token,
    safe_filename_stem as safe_filename_stem,
    unique_compact_export_stems as unique_compact_export_stems,
    unique_filename_stems as unique_filename_stems,
)

sys.modules[__name__] = _workflow
