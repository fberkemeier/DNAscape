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
    Every route reports its initial prediction. A numeric estimate starts only
    once prediction work has been measured.
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
        self._has_estimate = False
        self._finished: tuple[str, float] | None = None

    def start_case(self, total_cases: int) -> None:
        with self._lock:
            self._total_cases = max(1, total_cases)
            self._case_started = self._clock()
            self._work_started = None
            self._fraction = 0.0
            self._stage_eta = None
            self._has_estimate = False

    def report(self, value: float, message: str) -> None:
        with self._lock:
            now = self._clock()
            value = float(value)
            if not math.isfinite(value):
                return
            # These are service lifecycle markers, not measured solver work.
            measured_iteration = re.search(r"\b(?:iteration\s+)?\d+/\d+\b", message)
            if value >= 0.98:
                # Fitting has returned, possibly below its iteration limit.
                # Unused iterations must not become a new preview-time ETA.
                self._stage_eta = (0.0, now)
                self._fraction = 1.0
                self._has_estimate = True
                return
            if value <= 0.04 and not measured_iteration:
                if value == 0.04 and self._work_started is None:
                    self._work_started = now
                return
            if self._work_started is None:
                self._work_started = now if self._case_started is None else self._case_started
            match = _EXPLICIT_ETA.search(message)
            if match:
                hours, minutes, seconds = map(int, match.groups())
                self._stage_eta = (hours * 3600 + minutes * 60 + seconds, now)
                self._has_estimate = True
            self._fraction = min(1.0, max(0.0, (value - 0.04) / 0.90))
            if 0 < self._fraction < 1 and now - self._work_started >= 0.1:
                self._has_estimate = True

    def complete_case(self) -> None:
        with self._lock:
            if self._case_started is not None:
                self._case_seconds.append(max(0.0, self._clock() - self._case_started))
            self._case_started = None
            self._work_started = None
            self._fraction = 0.0
            self._stage_eta = None
            self._has_estimate = False

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
                if work_elapsed >= 0.1:
                    remaining = work_elapsed * (1 - self._fraction) / self._fraction
            # A stage which overran its estimate must not appear complete.
            if remaining is not None and remaining > 0:
                return f"ETA ≈ {_duration(remaining)} · {elapsed}"
            if self._has_estimate:
                # Re-estimate from observed progress when a stage overruns;
                # never erase a measured ETA during refinement/finalization.
                if self._work_started is not None and 0 < self._fraction < 1:
                    remaining = (now - self._work_started) * (1 - self._fraction) / self._fraction
                return f"ETA ≈ {_duration(max(1.0, remaining or 0.0))} · {elapsed}"
            if self._case_seconds and self._case_started is not None:
                average = sum(self._case_seconds) / len(self._case_seconds)
                current_elapsed = now - self._case_started
                cases_left = self._total_cases - len(self._case_seconds)
                remaining = average * cases_left - current_elapsed
                if remaining > 0:
                    return f"ETA ≈ {_duration(remaining)} (batch) · {elapsed}"
            return f"ETA estimating… · {elapsed}"
