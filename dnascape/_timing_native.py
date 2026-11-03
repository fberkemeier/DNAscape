"""Optional native stationary timing; NumPy remains the portable fallback."""
import ctypes
import hashlib
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np

_LOCK = threading.Lock()
_ATTEMPTED = False
_LIBRARY = None
_DOUBLE = ctypes.POINTER(ctypes.c_double)
_INDEX = ctypes.POINTER(ctypes.c_int64)


def timing_library():
    global _ATTEMPTED, _LIBRARY
    if os.environ.get("DNASCAPE_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}:
        return None
    with _LOCK:
        if _ATTEMPTED:
            return _LIBRARY
        _ATTEMPTED = True
        from ._inverse_native import _compiler

        try:
            source = Path(__file__).with_name("_timing_points.c").read_bytes()
            identity = source + f"{sys.platform}:{platform.machine()}:{ctypes.sizeof(ctypes.c_void_p)}".encode()
            digest = hashlib.sha256(identity).hexdigest()[:24]
            owner = hashlib.sha256(f"{os.environ.get('USERNAME', os.environ.get('USER', 'local'))}:{sys.executable}".encode()).hexdigest()[:12]
            cache = Path(tempfile.gettempdir()) / f"dnascape-native-{owner}"
            cache.mkdir(mode=0o700, parents=True, exist_ok=True)
            output = cache / (f"timing-{digest}" + (".dll" if sys.platform == "win32" else ".so"))
            if not output.is_file():
                compiler = _compiler()
                if compiler is None:
                    return None
                with tempfile.TemporaryDirectory(prefix="build-timing-", dir=cache) as directory:
                    captured = Path(directory) / "timing.c"
                    captured.write_bytes(source)
                    built = Path(directory) / output.name
                    command = [compiler, "-O3", "-std=c99", "-ffp-contract=off", "-shared"]
                    command += ["-static-libgcc"] if sys.platform == "win32" else ["-fPIC"]
                    command += [str(captured), "-o", str(built), "-lm"]
                    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
                    subprocess.run(command, check=True, capture_output=True, timeout=15, **options)
                    try:
                        os.replace(built, output)
                    except OSError:
                        if not output.is_file():
                            raise
            library = ctypes.CDLL(str(output))
            library.stationary_timing_abi.argtypes = []
            library.stationary_timing_abi.restype = ctypes.c_uint
            if library.stationary_timing_abi() != 1:
                return None
            base = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_int]
            library.stationary_timing_points.argtypes = base + [_DOUBLE]*9 + [_INDEX, _DOUBLE]
            library.stationary_timing_points.restype = None
            library.stationary_timing_pullback.argtypes = base + [ctypes.c_int] + [_DOUBLE]*9 + [_INDEX] + [_DOUBLE]*8
            library.stationary_timing_pullback.restype = None
            _LIBRARY = library
        except (OSError, AttributeError, subprocess.SubprocessError):
            pass
        return _LIBRARY


def _arguments(model, times, receivers):
    arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in
              (model.rate, model.slow, model.y, model.mass, model.moment,
               model.yc, model.center_prefix[0], model.center_prefix[1], times)]
    indices = np.ascontiguousarray(receivers, dtype=np.int64)
    if indices.ndim != 1 or arrays[-1].shape != indices.shape or np.any(indices < 0) or np.any(indices >= model.n):
        raise ValueError("Timing receivers must match the selected spatial grid.")
    base = [model.n, len(indices), model.dx, model.cap, int(model.periodic)]
    pointers = [a.ctypes.data_as(_DOUBLE) for a in arrays]
    return arrays, indices, base, pointers


def points(library, model, times, receivers):
    arrays, indices, base, pointers = _arguments(model, times, receivers)
    result = np.empty(len(indices))
    library.stationary_timing_points(*base, *pointers, indices.ctypes.data_as(_INDEX), result.ctypes.data_as(_DOUBLE))
    return result


def pullback(library, model, times, receivers, weights, mass_weights, buffers, with_speed):
    arrays, indices, base, pointers = _arguments(model, times, receivers)
    weights, mass_weights = np.ascontiguousarray(weights), np.ascontiguousarray(mass_weights)
    if weights.shape != indices.shape or mass_weights.shape != indices.shape:
        raise ValueError("Timing adjoint weights must match the quadrature points.")
    library.stationary_timing_pullback(*base, int(with_speed), *pointers,
        indices.ctypes.data_as(_INDEX), weights.ctypes.data_as(_DOUBLE), mass_weights.ctypes.data_as(_DOUBLE),
        *(a.ctypes.data_as(_DOUBLE) for a in buffers))
