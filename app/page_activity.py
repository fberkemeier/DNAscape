"""Client-local workflow status shared by pages and the navigation drawer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from nicegui import context


@dataclass
class PageRun:
    """One run; its identity prevents a stopped job reporting completion later."""

    cancel: Callable[[], None] | None = None
    running: bool = True
    stopped: bool = False
    indicator: Literal["running", "complete"] | None = "running"


class PageActivity:
    def __init__(self) -> None:
        self._runs: dict[str, PageRun] = {}
        self.on_change: Callable[[], None] | None = None
        self.is_page_active: Callable[[str], bool] | None = None

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def indicator(self, slug: str) -> Literal["running", "complete"] | None:
        run = self._runs.get(slug)
        return run.indicator if run is not None else None

    def retain_page(self, slug: str) -> bool:
        """Keep job controls and unread results alive during navigation."""

        run = self._runs.get(slug)
        return run is not None and (run.running or run.indicator == "complete")

    def begin(self, slug: str, *, cancel: Callable[[], None] | None = None) -> PageRun:
        run = PageRun(cancel=cancel)
        self._runs[slug] = run
        self._notify()
        return run

    def finish(self, slug: str, run: PageRun, *, success: bool) -> None:
        if self._runs.get(slug) is not run:
            return
        run.running = False
        is_active = self.is_page_active is not None and self.is_page_active(slug)
        run.indicator = "complete" if success and not run.stopped and not is_active else None
        self._notify()

    def stop(self, slug: str) -> None:
        run = self._runs.get(slug)
        if run is None:
            return
        run.stopped = True
        run.indicator = None
        if run.running and run.cancel is not None:
            run.cancel()
        self._notify()

    def acknowledge(self, slug: str) -> None:
        run = self._runs.get(slug)
        if run is not None and run.indicator == "complete":
            run.indicator = None
            self._notify()

    def reset(self) -> None:
        for run in self._runs.values():
            if run.running and run.cancel is not None:
                run.cancel()
        self._runs.clear()
        self._notify()


def current_page_activity() -> PageActivity:
    """Return status belonging only to this browser client."""

    storage = context.client.storage
    activity = storage.get("dnascape_page_activity")
    if not isinstance(activity, PageActivity):
        activity = PageActivity()
        storage["dnascape_page_activity"] = activity
    return activity
