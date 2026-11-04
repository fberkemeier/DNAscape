"""Preload expensive workflow modules; render pages only when opened."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from importlib import import_module

_HOSTED_PAGES = ("mapping", "data_management", "simulation", "plotting", "cached_files", "resources")


def preload_workflow_pages() -> bool:
    """Warm shared modules before serving; per-visitor controls stay lazy.

    Render used to defer the scientific imports until the first menu click,
    blocking its event loop for seconds. These modules will be needed for the
    workflows anyway, so load one shared copy at startup when headroom allows.
    Local startup keeps its existing selective preloading.
    """
    from dnascape._resources import MIB, HostedResourceError, ensure_capacity, hosted

    modules = _HOSTED_PAGES if hosted() else ("mapping", "data_management")
    pending = tuple(name for name in modules if f"{__name__}.{name}" not in sys.modules)
    if not pending:
        return True
    if hosted():
        try:
            # Scientific imports added about 100 MiB in a fresh process. Keep
            # a conservative allowance in addition to the service's reserve.
            ensure_capacity(128 * MIB, "Preparing workflow modules")
        except HostedResourceError:
            logging.getLogger(__name__).info("Deferring workflow preloading: limited memory headroom")
            return False
    for module_name in pending:
        import_module(f"{__name__}.{module_name}")
    return True


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

__all__ = ["PAGE_RENDERERS", "preload_workflow_pages"]
