"""Small, dependency-free terminal feedback while the app imports and starts."""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import TextIO


class LaunchProgress:
    """Report completed startup stages, animating only interactive terminals."""

    def __init__(self, stream: TextIO | None = None, *, operation="Launching app", label="Loading interface") -> None:
        self.stream = sys.stdout if stream is None else stream
        try:
            self.interactive = self.stream.isatty()
        except (AttributeError, OSError, ValueError):
            self.interactive = False
        self._stop = threading.Event()
        # A terminal interrupt can finish startup while the main thread is
        # already updating its progress line.
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stage = 0
        self._label = label
        self._operation = operation
        self._started = 0.
        self._frame = 0
        self._width = 0
        self._finished = False
        self._disabled = self.stream is None

    def __enter__(self) -> LaunchProgress:
        self._started = time.monotonic()
        with self._lock:
            self._render()
        if self.interactive and not self._disabled:
            self._thread = threading.Thread(target=self._animate, name="dnascape-launch", daemon=True)
            self._thread.start()
        return self

    def _write(self, text: str) -> None:
        if self._disabled:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):
            self._disabled = True
            self._stop.set()

    def _render(self, *, final: bool = False) -> None:
        filled = 20*self._stage//3
        bar = "#"*filled + "-"*(20-filled)
        spinner = "" if final or not self.interactive else " " + "|/-\\"[self._frame % 4]
        elapsed = int(time.monotonic()-self._started)
        line = f"{self._operation} [{bar}] {self._label}{spinner} ({elapsed}s)"
        if self.interactive:
            self._write("\r" + line.ljust(self._width) + ("\n" if final else ""))
            self._width = max(self._width, len(line))
        else:
            self._write(line + "\n")

    def _animate(self) -> None:
        while not self._stop.wait(.12):
            with self._lock:
                if self._stop.is_set():
                    break
                self._frame += 1
                self._render()

    def update(self, stage: int, label: str) -> None:
        with self._lock:
            if self._finished:
                return
            if stage < self._stage or not 0 <= stage <= 2:
                raise ValueError("Startup stages must advance from zero to two.")
            self._stage, self._label = stage, label
            self._render()

    def _finish(self, label: str, *, ready: bool = False) -> bool:
        with self._lock:
            if self._finished:
                return False
            self._finished = True
            self._stop.set()
            if ready:
                self._stage = 3
            self._label = label
            self._render(final=True)
        if self._thread is not None:
            self._thread.join(timeout=.5)
        return True

    def ready(self, urls=()) -> None:
        if self._finish("Ready", ready=True) and urls:
            self._write(f"DNAscape ready at {', '.join(sorted(urls))}\n")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._finished:
            self._finish("Cancelled" if exc_type is KeyboardInterrupt else
                         "Launch failed" if exc_type is not None else "Stopped")


class ShutdownProgress(LaunchProgress):
    """Start terminal feedback on the first interrupt, before server cleanup."""

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__(stream, operation="Stopping app", label="Cleaning up")
        self._begun = False
        self._handlers = {}

    def __enter__(self) -> ShutdownProgress:
        return self

    def start(self) -> None:
        if not self._begun:
            self._begun = True
            super().__enter__()

    def watch_signals(self, launch_progress: LaunchProgress | None = None) -> None:
        """Wrap the server's installed handlers without replacing its cleanup."""
        if threading.current_thread() is not threading.main_thread():
            return
        signals = [signal.SIGINT]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for signum in signals:
            if signum in self._handlers:
                continue
            previous = signal.getsignal(signum)
            if not callable(previous):
                continue

            def interrupted(sig, frame, handler=previous):
                if launch_progress is not None:
                    launch_progress._finish("Cancelled")
                self.start()
                handler(sig, frame)

            self._handlers[signum] = (previous, interrupted)
            signal.signal(signum, interrupted)

    def finish(self) -> None:
        self.start()
        if self._finish("Done", ready=True):
            self._write("DNAscape stopped\n")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for signum, (previous, installed) in self._handlers.items():
            # Uvicorn normally restores its original handlers itself, before
            # re-raising Ctrl+C. Do not overwrite a handler it has restored.
            if signal.getsignal(signum) is installed:
                signal.signal(signum, previous)
        if self._begun and not self._finished:
            self._finish("Shutdown failed" if exc_type is not None else "Stopped")
