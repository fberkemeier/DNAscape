"""Compatibility imports for :mod:`dnascape.workflows.export_settings`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import export_settings as _workflow
from dnascape.workflows.export_settings import (
    CACHE_FILE_FORMAT_OPTIONS as CACHE_FILE_FORMAT_OPTIONS,
    COMPRESSION_OPTIONS as COMPRESSION_OPTIONS,
    DATA_FILE_FORMAT_OPTIONS as DATA_FILE_FORMAT_OPTIONS,
    DEFAULT_CONVERSION_MEMORY_LIMIT_MB as DEFAULT_CONVERSION_MEMORY_LIMIT_MB,
    ExportSettings as ExportSettings,
    MAX_CONVERSION_MEMORY_LIMIT_MB as MAX_CONVERSION_MEMORY_LIMIT_MB,
    get_export_settings as get_export_settings,
    normalize_compression as normalize_compression,
    reset_export_settings as reset_export_settings,
)

sys.modules[__name__] = _workflow
