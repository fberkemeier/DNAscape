"""Compatibility imports for :mod:`dnascape.workflows.bigwig`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys

from dnascape.workflows import bigwig as _workflow
from dnascape.workflows.bigwig import (
    BigWigDependencyError as BigWigDependencyError,
    BigWigMetadata as BigWigMetadata,
    inspect_bigwig as inspect_bigwig,
    read_bigwig_bins as read_bigwig_bins,
)

sys.modules[__name__] = _workflow
