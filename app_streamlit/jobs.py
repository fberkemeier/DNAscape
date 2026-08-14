"""Small, Streamlit-independent background jobs for scientific workflows.

Workers never call Streamlit and never redirect process-global stdout.  A page
owns the job object in session state and polls it from the main script thread.
This lets mapping and simulation continue while the user visits another page.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Callable
from uuid import uuid4


class JobCancelled(RuntimeError):
    """Raised cooperatively when a background workflow is cancelled."""


MAX_PENDING_JOB_MESSAGES = 64
MAX_MESSAGES_PER_POLL = 128


def ensure_project_import_path() -> None:
    """Make local package imports reliable inside Streamlit worker threads."""

    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    for import_path in (project_root, app_dir):
        import_path_text = str(import_path)
        if import_path_text not in sys.path:
            sys.path.insert(0, import_path_text)


@dataclass(frozen=True)
class JobContext:
    """Thread-safe progress and cancellation channel passed to a worker."""

    _messages: Queue
    _cancel_event: Event

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        """Stop at the next cooperative checkpoint after cancellation."""

        if self.cancelled:
            raise JobCancelled("The calculation was cancelled.")

    def report(self, value: float, text: str | None = None) -> None:
        """Publish bounded determinate progress without touching Streamlit."""

        self.raise_if_cancelled()
        progress = min(1.0, max(0.0, float(value)))
        message = ("progress", progress, str(text or "Working..."))
        try:
            self._messages.put_nowait(message)
        except Full:
            # Progress is transient.  Discard the oldest update rather than
            # allowing a chatty scientific loop to grow session memory while
            # its page is not being rendered.
            try:
                self._messages.get_nowait()
            except Empty:
                pass
            try:
                self._messages.put_nowait(message)
            except Full:
                pass

    def _publish_terminal(self, message: tuple[Any, ...]) -> None:
        """Publish a terminal message even when progress filled the queue."""

        while True:
            try:
                self._messages.put_nowait(message)
                return
            except Full:
                try:
                    self._messages.get_nowait()
                except Empty:
                    continue


@dataclass
class BackgroundJob:
    """Mutable state for one explicitly requested background calculation."""

    label: str
    input_signature: str
    job_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"
    result: Any = None
    error_message: str | None = None
    error_type: str | None = None
    consumed: bool = False
    progress_value: float = 0.02
    progress_text: str = "Starting..."
    cancellation_requested: bool = False
    _messages: Queue = field(
        default_factory=lambda: Queue(maxsize=MAX_PENDING_JOB_MESSAGES), repr=False
    )
    _cancel_event: Event = field(default_factory=Event, repr=False)
    _thread: Thread | None = field(default=None, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed wall time without blocking the worker."""

        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, float(end - self.started_at))

    @property
    def running(self) -> bool:
        return self.status == "running"


def _run_job(
    context: JobContext,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        ensure_project_import_path()
        result = function(*args, job_context=context, **kwargs)
        context.raise_if_cancelled()
        context._publish_terminal(("complete", result))
    except JobCancelled:
        context._publish_terminal(("cancelled",))
    except Exception as exc:  # The main thread formats the actionable error.
        if context.cancelled:
            context._publish_terminal(("cancelled",))
        else:
            context._publish_terminal(("error", type(exc).__name__, str(exc)))


def start_background_job(
    label: str,
    input_signature: str,
    function: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> BackgroundJob:
    """Start one daemon worker and return its page-owned state object."""

    job = BackgroundJob(label=str(label), input_signature=str(input_signature))
    context = JobContext(job._messages, job._cancel_event)
    job._thread = Thread(
        target=_run_job,
        args=(context, function, args, kwargs),
        name=f"dnascape-{job.job_id[:10]}",
        daemon=True,
    )
    job._thread.start()
    return job


def cancel_background_job(job: BackgroundJob) -> bool:
    """Request cooperative cancellation and keep polling until acknowledged."""

    if not isinstance(job, BackgroundJob) or not job.running:
        return False
    job.cancellation_requested = True
    job.progress_text = "Cancelling..."
    job._cancel_event.set()
    return True


def poll_background_job(job: BackgroundJob) -> BackgroundJob:
    """Drain all currently available progress and terminal messages."""

    if not isinstance(job, BackgroundJob) or not job.running:
        return job
    processed = 0
    while job.running and processed < MAX_MESSAGES_PER_POLL:
        try:
            message = job._messages.get_nowait()
        except Empty:
            break
        processed += 1

        status = message[0]
        if status == "progress":
            job.progress_value = min(0.99, max(job.progress_value, float(message[1])))
            job.progress_text = str(message[2])
            continue

        job.finished_at = time.time()
        if status == "complete":
            if job.cancellation_requested:
                job.status = "cancelled"
                job.progress_text = "Cancelled"
                job.result = None
            else:
                job.status = "complete"
                job.progress_value = 1.0
                job.progress_text = "Complete"
                job.result = message[1]
        elif status == "cancelled":
            job.status = "cancelled"
            job.progress_text = "Cancelled"
        else:
            job.status = "error"
            job.error_type = str(message[1])
            job.error_message = str(message[2]) or "The calculation failed."

    # A worker should always publish a terminal message.  Detect an abnormal
    # thread exit so a dead job cannot leave every execution button disabled
    # forever.
    if (
        job.running
        and job._thread is not None
        and not job._thread.is_alive()
        and job._messages.empty()
    ):
        job.finished_at = time.time()
        job.status = "error"
        job.error_type = "WorkerExited"
        job.error_message = "The background worker exited without returning a result."
    return job


def any_background_job_running(*jobs: Any) -> bool:
    """Poll known jobs and report whether any one is still active."""

    running = False
    for job in jobs:
        if not isinstance(job, BackgroundJob):
            continue
        poll_background_job(job)
        running = running or job.running
    return running
