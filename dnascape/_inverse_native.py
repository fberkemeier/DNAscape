"""Optional, dependency-free native acceleration for inverse fork transport.

The bundled C source is compiled once into a source/platform-keyed temporary
cache when a GCC-compatible compiler is available. No binary is distributed,
and failure to compile/load simply selects the NumPy implementation. Set
``DNASCAPE_DISABLE_NATIVE=1`` to force that implementation. This kernel keeps
the same grid, CFL subdivisions, clipping and exact parameter sensitivities;
it does not downsample data or change the fitted model.
"""

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from .utils import check_cancelled

_LOCK = threading.Lock()
_LIBRARY = None
_ATTEMPTED = False
_UNAVAILABLE_REASON = None
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
_ABI_VERSION = 2


def _compiler():
    for name in ("gcc", "clang", "cc"):
        found = shutil.which(name)
        if found:
            return found
    # Strawberry Perl commonly supplies GCC on Windows without adding it to
    # the application's PATH. This is a compiler candidate, not a dependency.
    if sys.platform == "win32":
        candidate = Path("C:/Strawberry/c/bin/gcc.exe")
        if candidate.is_file():
            return str(candidate)
    return None


def _load_library():
    global _ATTEMPTED, _LIBRARY, _UNAVAILABLE_REASON
    if os.environ.get("DNASCAPE_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}:
        return None
    with _LOCK:
        if _ATTEMPTED:
            return _LIBRARY
        _ATTEMPTED = True
        try:
            source = Path(__file__).with_name("_inverse_transport.c")
            source_bytes = source.read_bytes()
            identity = (
                source_bytes
                + f"{sys.platform}:{platform.machine()}:{ctypes.sizeof(ctypes.c_void_p)}".encode()
            )
            digest = hashlib.sha256(identity).hexdigest()[:24]
            user = str(getattr(os, "getuid", lambda: os.environ.get("USERNAME", "local"))())
            user_digest = hashlib.sha256(f"{user}:{sys.executable}".encode()).hexdigest()[:12]
            cache = Path(tempfile.gettempdir()) / f"dnascape-native-{user_digest}"
            cache.mkdir(mode=0o700, parents=True, exist_ok=True)
            suffix = ".dll" if sys.platform == "win32" else ".so"
            library_path = cache / f"inverse-{digest}{suffix}"
            if not library_path.is_file():
                compiler = _compiler()
                if compiler is None:
                    _UNAVAILABLE_REASON = "No supported C compiler is available."
                    return None
                # Each process compiles separately, then atomically publishes;
                # readers never see a partial library from another process.
                with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as temporary:
                    output = Path(temporary) / f"inverse{suffix}"
                    # Compile exactly the bytes used for the cache key even
                    # if the checkout changes during an application update.
                    captured_source = Path(temporary) / "inverse_transport.c"
                    captured_source.write_bytes(source_bytes)
                    command = [compiler, "-O3", "-std=c99", "-ffp-contract=off", "-shared"]
                    if sys.platform == "win32":
                        command += ["-static-libgcc"]
                    else:
                        command += ["-fPIC"]
                    command += [str(captured_source), "-o", str(output)]
                    options = {}
                    if sys.platform == "win32":
                        options["creationflags"] = subprocess.CREATE_NO_WINDOW
                    subprocess.run(command, check=True, capture_output=True, timeout=15, **options)
                    try:
                        os.replace(output, library_path)
                    except OSError:
                        if not library_path.is_file():
                            raise
            library = ctypes.CDLL(str(library_path))
            abi_version = library.inverse_transport_abi_version
            abi_version.argtypes = []
            abi_version.restype = ctypes.c_uint
            if abi_version() != _ABI_VERSION:
                _UNAVAILABLE_REASON = "Native transport ABI does not match the Python wrapper."
                return None
            # The versioned symbol also protects an older live Python process
            # from calling a new source file's incompatible signature.
            function = library.inverse_transport_interval_v2
            function.argtypes = (
                [ctypes.c_size_t] * 5
                + [ctypes.c_double] * 2
                + [ctypes.c_int]
                + [ctypes.c_size_t] * 4
                + [_DOUBLE_POINTER] * 17
            )
            function.restype = None
            _LIBRARY = library
        except (OSError, AttributeError, subprocess.SubprocessError) as error:
            _UNAVAILABLE_REASON = str(error)
        return _LIBRARY


def create_transport(n, p):
    """Return a reusable interval kernel, or None for the NumPy fallback."""
    library = _load_library()
    return None if library is None else _Transport(library, n, p)


class _Transport:
    def __init__(self, library, n, p):
        if n < 1 or p < 0:
            raise ValueError("Transport requires positive positions and nonnegative parameters.")
        self.n, self.p = int(n), int(p)
        self._function = library.inverse_transport_interval_v2
        self._scratch = np.empty(3 * self.n * (self.p + 1), dtype=np.float64)

    def advance(
        self, interval, substeps, resolution, perQ,
        i0, i1, v0, v1, di0, di1, dv0, dv1,
        f, right, left, timing, df, dright, dleft, dtiming,
    ):
        """Advance mutually disjoint contiguous float64 states over one interval.

        Field endpoints can contain one value or n positions. Sensitivities
        have shape (n, p). None sensitivities are allowed when p is zero.
        Cancellation is also checked between bounded batches of CFL substeps,
        so an unusually large explicit time interval cannot monopolize Stop.
        """
        if interval <= 0 or resolution <= 0 or substeps < 1:
            raise ValueError("Interval, resolution and substeps must be positive.")
        values = [np.ascontiguousarray(a, dtype=np.float64).reshape(-1) for a in (i0, i1, v0, v1)]
        if any(a.size not in (1, self.n) for a in values):
            raise ValueError("Field endpoints must be scalar or have one value per position.")
        strides = [int(a.size != 1) for a in values]
        if self.p:
            derivatives = [np.ascontiguousarray(a, dtype=np.float64) for a in (di0, di1, dv0, dv1)]
            if any(a.shape != (self.n, self.p) for a in derivatives):
                raise ValueError("Field sensitivities must have shape (positions, parameters).")
        else:
            derivatives = [self._scratch[:0]] * 4
        state = [f, right, left, timing, df, dright, dleft, dtiming]
        for index, array in enumerate(state):
            shape = (self.n,) if index < 4 else (self.n, self.p)
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float64
                or array.shape != shape
                or not array.flags.c_contiguous
                or not array.flags.writeable
            ):
                raise ValueError("Native state must be writable contiguous float64 with matching shape.")
        pointers = [
            a.ctypes.data_as(_DOUBLE_POINTER)
            for a in values + derivatives + state + [self._scratch]
        ]
        # Bound both substep count and position/parameter work per C call.
        # Global substep indices preserve midpoint weights; the C kernel
        # saves/integrates the trapezoid only at the interval's true endpoints.
        chunk = max(1, min(128, 1_000_000 // (self.n * (self.p + 1))))
        substeps = int(substeps)
        for first_step in range(0, substeps, chunk):
            check_cancelled()
            self._function(
                self.n, self.p, substeps, first_step, min(chunk, substeps - first_step),
                float(interval), float(resolution), bool(perQ), *strides, *pointers,
            )
