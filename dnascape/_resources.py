"""Render-only resource policy; local workflows retain their existing limits.

Estimates are checked before allocation and cooperative checkpoints protect
growing jobs. This is a safety margin, not a replacement for OS isolation.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import RLock, local

MIB = 1024**2
LOCAL_SETUP_URL = "https://github.com/fberkemeier/DNAscape#run-the-dnascape-app"
_LOCK = RLock()
_STATE = local()


class HostedResourceError(ValueError):
    """A hosted job was declined or stopped before exhausting its budget."""


def hosted() -> bool:
    return os.environ.get("RENDER", "").strip().lower() == "true"


def _read_number(path):
    try:
        value = int(Path(path).read_text().strip())
        return value if 0 < value < 2**60 else None
    except (OSError, ValueError):
        return None


def _cgroup_pairs():
    # Include nested cgroups on hosts without a private cgroup namespace.
    roots = [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            base = Path("/sys/fs/cgroup")
            if "memory" in controllers.split(","):
                base /= "memory"
                names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
            elif controllers == "":
                names = ("memory.max", "memory.current")
            else:
                continue
            # Reject '..' from the kernel path before joining.
            if ".." not in Path(relative).parts:
                base /= relative.lstrip("/")
                roots.append(tuple(str(base / name) for name in names))
    except (OSError, ValueError):
        pass
    return roots


@dataclass(frozen=True)
class MemoryStatus:
    limit: int
    used: int
    reserve: int

    @property
    def available(self):
        return max(0, self.limit - self.used - self.reserve)


def memory_status() -> MemoryStatus | None:
    if not hosted():
        return None
    measurements = []
    for limit_path, usage_path in _cgroup_pairs():
        limit, used = _read_number(limit_path), _read_number(usage_path)
        if limit is not None:
            measurements.append((limit, used))
    limit, used = min(measurements, default=(512 * MIB, None), key=lambda x: x[0])
    # An explicit budget can lower the detected limit, never raise it.
    try:
        configured = float(os.environ.get("DNASCAPE_MEMORY_LIMIT_MB", "nan"))
        if math.isfinite(configured) and configured > 0:
            limit = min(limit, int(configured * MIB))
    except ValueError:
        pass
    if used is None:
        try:
            used = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            # Conservative fallback when neither container nor process usage
            # is readable. Do not pretend the web app consumes no memory.
            used = limit // 2
    return MemoryStatus(limit, used, min(limit // 3, max(64 * MIB, limit // 6)))


def resource_message(label="This job"):
    return (f"Online memory limit: {label} needs more memory than is safely available. "
            "Use a smaller region, larger bin size or fewer outputs, or run DNAscape "
            "locally for larger jobs. Your inputs and settings are unchanged.")


def ensure_capacity(additional_bytes=0, label="This job"):
    status = memory_status()
    if status is not None and max(0, int(additional_bytes)) >= status.available:
        details = (f" The next step needs about {additional_bytes / MIB:.0f} MiB; "
                   f"{status.available / MIB:.0f} MiB is safely available."
                   if additional_bytes else "")
        raise HostedResourceError(resource_message(label) + details)


def checkpoint():
    if not hosted():
        return
    now = time.monotonic()
    if now - getattr(_STATE, "checked", 0.0) >= .1:
        _STATE.checked = now
        ensure_capacity(label="The running job")


def working_bytes(maximum=32 * MIB):
    status = memory_status()
    return maximum if status is None else max(0, min(maximum, status.available // 3))


def upload_limit():
    status = memory_status()
    return None if status is None else max(1, min(16 * MIB, status.available // 8))


@contextmanager
def operation(label):
    if not hosted():
        yield
        return
    if not _LOCK.acquire(blocking=False):
        raise HostedResourceError("Online app busy: another data or calculation job is running. "
                                  "Try again when it finishes, or run DNAscape locally.")
    handle = None
    depth = getattr(_STATE, "depth", 0)
    try:
        if depth == 0 and os.name == "posix":
            import fcntl
            handle = open(Path(tempfile.gettempdir()) / "dnascape-hosted-job.lock", "a+b")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HostedResourceError("Online app busy: another data or calculation job "
                                          "is running. Try again when it finishes.") from exc
        _STATE.depth = depth + 1
        ensure_capacity(label=label)
        yield
        ensure_capacity(label=label)
    except MemoryError as exc:
        raise HostedResourceError(resource_message(label)) from exc
    finally:
        _STATE.depth = depth
        if handle is not None:
            handle.close()
        _LOCK.release()


def hosted_operation(label):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with operation(label):
                try:
                    return function(*args, **kwargs)
                except HostedResourceError:
                    raise
                except ValueError as exc:
                    if hosted() and any(word in str(exc).lower() for word in (
                        "memory limit", "too large", "preallocate", "larger than", "safe inspection limit",
                    )):
                        raise HostedResourceError(resource_message(label)) from exc
                    raise
        return wrapped
    return decorate
