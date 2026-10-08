"""Compatibility imports for :mod:`dnascape.workflows.mapping`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import mapping as _workflow
from dnascape.workflows.mapping import (
    CancelChecker as CancelChecker,
    MAX_GRID_CELLS as MAX_GRID_CELLS,
    MAX_INPUT_BYTES as MAX_INPUT_BYTES,
    MAX_INPUT_VALUES as MAX_INPUT_VALUES,
    MAX_MAPPING_IOD_SAMPLES as MAX_MAPPING_IOD_SAMPLES,
    MAX_MAPPING_SIMULATIONS as MAX_MAPPING_SIMULATIONS,
    MAX_OUTPUT_BYTES as MAX_OUTPUT_BYTES,
    MAX_OUTPUT_VALUES as MAX_OUTPUT_VALUES,
    MAX_SPATIAL_POINTS as MAX_SPATIAL_POINTS,
    MAX_TIME_POINTS as MAX_TIME_POINTS,
    MAX_TRACK_FILES as MAX_TRACK_FILES,
    MAX_TRACK_PAYLOAD_BYTES as MAX_TRACK_PAYLOAD_BYTES,
    MAX_TRACK_UPLOAD_BYTES as MAX_TRACK_UPLOAD_BYTES,
    MappingComparison as MappingComparison,
    MappingRequest as MappingRequest,
    MappingResult as MappingResult,
    ProgressReporter as ProgressReporter,
    TrackUpload as TrackUpload,
    materialize_many_tracks as materialize_many_tracks,
    materialize_track_upload as materialize_track_upload,
    run_mapping as run_mapping,
    validate_mapping_request as validate_mapping_request,
)

sys.modules[__name__] = _workflow
