"""Load workflow pages when opened instead of during server startup."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module


def _lazy_renderer(module_name: str) -> Callable[[Callable[[str], None]], None]:
    def render(navigate: Callable[[str], None]) -> None:
        import_module(f"{__name__}.{module_name}").render(navigate)

    return render

PAGE_RENDERERS = {
    "overview": _lazy_renderer("overview"),
    "data-management": _lazy_renderer("data_management"),
    "mapping": _lazy_renderer("mapping"),
    "simulation": _lazy_renderer("simulation"),
    "plotting": _lazy_renderer("plotting"),
    "cached-files": _lazy_renderer("cached_files"),
    "resources": _lazy_renderer("resources"),
}

__all__ = ["PAGE_RENDERERS"]
