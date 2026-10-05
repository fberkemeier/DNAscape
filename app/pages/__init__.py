"""Page registry for the DNAscape NiceGUI prototype."""

from __future__ import annotations

from app.pages import (
    cached_files,
    data_management,
    mapping,
    overview,
    plotting,
    resources,
    simulation,
)

PAGE_RENDERERS = {
    "overview": overview.render,
    "data-management": data_management.render,
    "mapping": mapping.render,
    "simulation": simulation.render,
    "plotting": plotting.render,
    "cached-files": cached_files.render,
    "resources": resources.render,
}

__all__ = ["PAGE_RENDERERS"]
