"""Workflow settings API with client resolution installed by the app layer."""

from __future__ import annotations

import sys

from app.preferences import client_settings
from dnascape.workflows import export_settings as _workflow
from dnascape.workflows.export_settings import (
    CACHE_FILE_FORMAT_OPTIONS as CACHE_FILE_FORMAT_OPTIONS,
)
from dnascape.workflows.export_settings import (
    COMPRESSION_OPTIONS as COMPRESSION_OPTIONS,
)
from dnascape.workflows.export_settings import (
    DATA_FILE_FORMAT_OPTIONS as DATA_FILE_FORMAT_OPTIONS,
)
from dnascape.workflows.export_settings import (
    DEFAULT_CONTROL_FONT_SIZE_PX as DEFAULT_CONTROL_FONT_SIZE_PX,
)
from dnascape.workflows.export_settings import (
    DEFAULT_CONVERSION_MEMORY_LIMIT_MB as DEFAULT_CONVERSION_MEMORY_LIMIT_MB,
)
from dnascape.workflows.export_settings import (
    MAX_CONTROL_FONT_SIZE_PX as MAX_CONTROL_FONT_SIZE_PX,
)
from dnascape.workflows.export_settings import (
    MAX_CONVERSION_MEMORY_LIMIT_MB as MAX_CONVERSION_MEMORY_LIMIT_MB,
)
from dnascape.workflows.export_settings import (
    MIN_CONTROL_FONT_SIZE_PX as MIN_CONTROL_FONT_SIZE_PX,
)
from dnascape.workflows.export_settings import (
    ExportSettings as ExportSettings,
)
from dnascape.workflows.export_settings import (
    get_export_settings as get_export_settings,
)
from dnascape.workflows.export_settings import (
    normalize_compression as normalize_compression,
)
from dnascape.workflows.export_settings import (
    reset_export_settings as reset_export_settings,
)

_workflow.set_settings_provider(client_settings)
sys.modules[__name__] = _workflow
