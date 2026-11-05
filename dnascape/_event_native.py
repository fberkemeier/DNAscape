"""Optional acceleration of event-only transport; NumPy is the fallback."""

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from .utils import check_cancelled

_LOCK = threading.Lock()
_FUNCTION = None
_ATTEMPTED = False
_DOUBLE = ctypes.POINTER(ctypes.c_double)


def create_transport(n):
    global _FUNCTION, _ATTEMPTED
    if os.environ.get("DNASCAPE_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}:
        return None
    with _LOCK:
        if not _ATTEMPTED:
            _ATTEMPTED = True
            try:
                from ._inverse_native import _compiler

                source = Path(__file__).with_name("_event_transport.c").read_bytes()
                identity = source + f"{sys.platform}:{platform.machine()}:{sys.executable}".encode()
                key = hashlib.sha256(identity).hexdigest()[:24]
                directory = Path(tempfile.gettempdir()) / "dnascape-event-native"
                directory.mkdir(mode=0o700, exist_ok=True)
                suffix = ".dll" if sys.platform == "win32" else ".so"
                target = directory / (key + suffix)
                if not target.is_file():
                    compiler = _compiler()
                    if compiler is None:
                        return None
                    with tempfile.TemporaryDirectory(dir=directory) as temporary:
                        captured = Path(temporary) / "events.c"
                        output = Path(temporary) / ("events" + suffix)
                        captured.write_bytes(source)
                        command = [compiler, "-O3", "-std=c99", "-ffp-contract=off", "-shared"]
                        command += ["-static-libgcc"] if sys.platform == "win32" else ["-fPIC"]
                        command += [str(captured), "-o", str(output)]
                        if sys.platform != "win32":
                            command += ["-lm"]
                        options = (
                            {"creationflags": subprocess.CREATE_NO_WINDOW}
                            if sys.platform == "win32"
                            else {}
                        )
                        subprocess.run(
                            command, check=True, capture_output=True, timeout=15, **options
                        )
                        try:
                            os.replace(output, target)
                        except OSError:
                            if not target.is_file():
                                raise
                library = ctypes.CDLL(str(target))
                function = library.event_interval
                function.argtypes = (
                    [ctypes.c_size_t] * 4
                    + [ctypes.c_double] * 2
                    + [ctypes.c_int] * 2
                    + [ctypes.c_double]
                    + [ctypes.c_int] * 2
                    + [_DOUBLE] * 6
                )
                function.restype = None
                function._library = library
                _FUNCTION = function
            except (OSError, AttributeError, subprocess.SubprocessError):
                return None
    return None if _FUNCTION is None else _Transport(n, _FUNCTION)


class _Transport:
    def __init__(self, n, function):
        self.n, self.function = n, function
        self.work = np.empty(26 * n)

    def advance(self, state, interval, steps, dx, periodic, counts, floor, i0, i1, v0, v1):
        endpoints = [np.ascontiguousarray(a, dtype=float).reshape(-1) for a in (i0, i1, v0, v1)]
        chunk = max(1, min(64, 500_000 // self.n))
        pointers = [a.ctypes.data_as(_DOUBLE) for a in endpoints + [state, self.work]]
        for first in range(0, steps, chunk):
            check_cancelled()
            self.function(
                self.n,
                steps,
                first,
                min(chunk, steps - first),
                interval,
                dx,
                periodic,
                counts,
                floor,
                int(endpoints[0].size > 1),
                int(endpoints[2].size > 1),
                *pointers,
            )
