"""Compatibility imports for :mod:`dnascape.workflows.mapping_exports`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import mapping_exports as _workflow
from dnascape.workflows.mapping_exports import (
    MAX_BINARY_EXPORT_BYTES as MAX_BINARY_EXPORT_BYTES,
    MAX_TEXT_EXPORT_VALUES as MAX_TEXT_EXPORT_VALUES,
    SelectedMappingOutput as SelectedMappingOutput,
    build_mapping_comparison_raw_export as build_mapping_comparison_raw_export,
    build_mapping_raw_export as build_mapping_raw_export,
    mapping_output_dependence_suffix as mapping_output_dependence_suffix,
    select_mapping_output as select_mapping_output,
)

sys.modules[__name__] = _workflow
