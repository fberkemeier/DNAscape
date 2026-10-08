"""Compatibility imports for :mod:`dnascape.workflows.simulation`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import simulation as _workflow
from dnascape.workflows.simulation import (
    CORE_OUTPUT_KEYS as CORE_OUTPUT_KEYS,
    CancelChecker as CancelChecker,
    ConstantSimulationRequest as ConstantSimulationRequest,
    ConstantSimulationResult as ConstantSimulationResult,
    DETAILED_XT_ARRAY_COUNT as DETAILED_XT_ARRAY_COUNT,
    DETAILED_XT_ITEM_BYTES as DETAILED_XT_ITEM_BYTES,
    DISTRIBUTION_OUTPUT_KEYS as DISTRIBUTION_OUTPUT_KEYS,
    MAX_DETAILED_XT_BYTES as MAX_DETAILED_XT_BYTES,
    MAX_DETAILED_XT_CELLS as MAX_DETAILED_XT_CELLS,
    MAX_INTER_ORIGIN_SAMPLES as MAX_INTER_ORIGIN_SAMPLES,
    MAX_REPLICATES as MAX_REPLICATES,
    MAX_RETAINED_REPLICATION_PROFILES as MAX_RETAINED_REPLICATION_PROFILES,
    MAX_RETAINED_REPLICATION_PROFILE_BYTES as MAX_RETAINED_REPLICATION_PROFILE_BYTES,
    MAX_SPATIAL_BINS as MAX_SPATIAL_BINS,
    MAX_TIME_POINTS as MAX_TIME_POINTS,
    POOLED_DISTRIBUTION_OUTPUT_KEYS as POOLED_DISTRIBUTION_OUTPUT_KEYS,
    ProgressReporter as ProgressReporter,
    REPLICATE_OUTPUT_KEYS as REPLICATE_OUTPUT_KEYS,
    TIME_OUTPUT_KEYS as TIME_OUTPUT_KEYS,
    XT_OUTPUT_KEYS as XT_OUTPUT_KEYS,
    run_constant_simulation as run_constant_simulation,
    validate_constant_simulation as validate_constant_simulation,
)

sys.modules[__name__] = _workflow
