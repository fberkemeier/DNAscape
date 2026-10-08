"""Compatibility imports for :mod:`dnascape.workflows.mapping_catalog`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import mapping_catalog as _workflow
from dnascape.workflows.mapping_catalog import (
    DATA_FORMS as DATA_FORMS,
    DEPENDENCE_ORDER as DEPENDENCE_ORDER,
    DataForm as DataForm,
    InputRole as InputRole,
    MappingCatalog as MappingCatalog,
    MappingInput as MappingInput,
    MappingRoute as MappingRoute,
    MappingRouteFamily as MappingRouteFamily,
    MappingSetting as MappingSetting,
    MappingTarget as MappingTarget,
    RDATA_FORMATS as RDATA_FORMATS,
    RDATA_FORM_HINT as RDATA_FORM_HINT,
    RDATA_PLAIN_FORMAT_KEYS as RDATA_PLAIN_FORMAT_KEYS,
    RDATA_PLAIN_UPLOAD_ACCEPT as RDATA_PLAIN_UPLOAD_ACCEPT,
    RDATA_SOURCE_FORMAT_KEYS as RDATA_SOURCE_FORMAT_KEYS,
    RDATA_TARGET_FORMAT_KEYS as RDATA_TARGET_FORMAT_KEYS,
    RDATA_TRACK_FORMAT_KEYS as RDATA_TRACK_FORMAT_KEYS,
    RDATA_TRACK_UPLOAD_ACCEPT as RDATA_TRACK_UPLOAD_ACCEPT,
    RDATA_UPLOAD_ACCEPT as RDATA_UPLOAD_ACCEPT,
    RDATA_UPLOAD_HINT as RDATA_UPLOAD_HINT,
    RDataFormat as RDataFormat,
    SettingCategory as SettingCategory,
    SettingKind as SettingKind,
    SettingLocation as SettingLocation,
    T as T,
    TargetVariant as TargetVariant,
    TrackUploadMode as TrackUploadMode,
    get_mapping_catalog as get_mapping_catalog,
    mapping_route as mapping_route,
    mapping_route_families as mapping_route_families,
    mapping_route_family as mapping_route_family,
    mapping_routes as mapping_routes,
    mapping_target as mapping_target,
    mapping_targets as mapping_targets,
    rdata_formats as rdata_formats,
)

sys.modules[__name__] = _workflow
