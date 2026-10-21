"""Compatibility imports for :mod:`dnascape.workflows.conversion`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import conversion as _workflow
from dnascape.workflows.filtering import DataFilterSettings as DataFilterSettings
from dnascape.workflows.conversion import (
    DATA_EXPORT_FORMATS as DATA_EXPORT_FORMATS,
    DEFAULT_TARGET_FORMAT as DEFAULT_TARGET_FORMAT,
    DataConversionRequest as DataConversionRequest,
    DataConversionResult as DataConversionResult,
    DataPreview as DataPreview,
    DataSourceInspection as DataSourceInspection,
    DataValueArray as DataValueArray,
    PreparedData as PreparedData,
    TrackInspection as TrackInspection,
    TrackRegion as TrackRegion,
    convert_data as convert_data,
    extract_values as extract_values,
    inspect_data_source as inspect_data_source,
    inspect_track as inspect_track,
    prepare_data as prepare_data,
    transcode_conversion_artifact as transcode_conversion_artifact,
)

sys.modules[__name__] = _workflow
