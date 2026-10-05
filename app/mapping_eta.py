"""Honest, continuously refreshed estimates for Mapping worker progress."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from threading import Lock
from time import perf_counter

_EXPLICIT_ETA = re.compile(r"\bETA\s*:?\s*(\d+):(\d{2}):(\d{2})", re.IGNORECASE)


def _duration(seconds: float) -> str:
    total = max(0, math.ceil(seconds))
    hours, remaining = divmod(total, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class MappingEta:
    """Estimate actual work, never extrapolate the decorative preparation bar.

    Solver ETA describes its current stage. A batch estimate based on completed
    cases is identified separately because cases need not cost the same amount.
    Quiet routes remain explicitly unestimated until a case has completed.
    """

    def __init__(self, *, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock
        self._lock = Lock()
        self._started = clock()
        self._case_started: float | None = None
        self._work_started: float | None = None
        self._case_seconds: list[float] = []
        self._total_cases = 1
        self._fraction = 0.0
        self._stage_eta: tuple[float, float] | None = None
        self._finished: tuple[str, float] | None = None

    def start_case(self, total_cases: int) -> None:
        with self._lock:
            self._total_cases = max(1, total_cases)
            self._case_started = self._clock()
            self._work_started = None
            self._fraction = 0.0
            self._stage_eta = None

    def report(self, value: float, message: str) -> None:
        with self._lock:
            now = self._clock()
            value = float(value)
            if not math.isfinite(value):
                return
            # These are service lifecycle markers, not measured solver work.
            if value <= .04 or value >= .98:
                self._stage_eta = None
                self._fraction = 0.0
                if value == .04:
                    self._work_started = now
                return
            if self._work_started is None:
                self._work_started = (
                    now if self._case_started is None else self._case_started
                )
            match = _EXPLICIT_ETA.search(message)
            if match:
                hours, minutes, seconds = map(int, match.groups())
                self._stage_eta = (hours * 3600 + minutes * 60 + seconds, now)
            else:
                self._stage_eta = None
            self._fraction = min(1.0, max(0.0, (value - .04) / .90))

    def complete_case(self) -> None:
        with self._lock:
            if self._case_started is not None:
                self._case_seconds.append(max(0.0, self._clock() - self._case_started))
            self._case_started = None
            self._work_started = None
            self._fraction = 0.0
            self._stage_eta = None

    def finish(self, outcome: str) -> None:
        with self._lock:
            self._finished = (outcome, self._clock())

    def text(self) -> str:
        with self._lock:
            now = self._clock() if self._finished is None else self._finished[1]
            elapsed = f"Elapsed {_duration(now - self._started)}"
            if self._finished is not None:
                outcome = self._finished[0]
                eta = "00:00:00" if outcome == "complete" else outcome
                return f"ETA {eta} · {elapsed}"
            remaining: float | None = None
            if self._stage_eta is not None:
                seconds, reported = self._stage_eta
                remaining = seconds - (now - reported)
            elif self._work_started is not None and 0 < self._fraction < 1:
                work_elapsed = now - self._work_started
                if work_elapsed >= .1:
                    remaining = work_elapsed * (1 - self._fraction) / self._fraction
            # A stage which overran its estimate must not appear complete.
            if remaining is not None and remaining > 0:
                return f"ETA ≈ {_duration(remaining)} (current stage) · {elapsed}"
            if self._case_seconds and self._case_started is not None:
                average = sum(self._case_seconds) / len(self._case_seconds)
                current_elapsed = now - self._case_started
                cases_left = self._total_cases - len(self._case_seconds)
                remaining = average * cases_left - current_elapsed
                if remaining > 0:
                    return f"ETA ≈ {_duration(remaining)} (batch) · {elapsed}"
            return f"ETA estimating… · {elapsed}"
