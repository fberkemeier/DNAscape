"""Compatibility imports for :mod:`dnascape.workflows.batch`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import batch as _workflow
from dnascape.workflows.batch import (
    BatchPlan as BatchPlan,
    BatchSourceSummary as BatchSourceSummary,
    CombinationMode as CombinationMode,
    CombinationSource as CombinationSource,
    T as T,
    batch_item as batch_item,
    combination_names as combination_names,
    dataset_stem as dataset_stem,
    named_batch as named_batch,
    plan_batch_sources as plan_batch_sources,
    plan_combination_indices as plan_combination_indices,
    target_first_combination_names as target_first_combination_names,
    unique_dataset_names as unique_dataset_names,
)

sys.modules[__name__] = _workflow
