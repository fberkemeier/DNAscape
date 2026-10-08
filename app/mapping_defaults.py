"""Compatibility imports for :mod:`dnascape.workflows.mapping_defaults`."""

from __future__ import annotations

import sys

from dnascape.workflows import mapping_defaults as _workflow
from dnascape.workflows.mapping_defaults import mapping_defaults as mapping_defaults

sys.modules[__name__] = _workflow
