"""Native transport preserves the NumPy model, boundaries and sensitivities."""

import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from dnascape import _inverse_native
from dnascape.inverse import _make_field, _mean_field_timing


def run_native(firing, speed, t, resolution, periodic, fields=None, parameters=None):
    n, p = firing.shape[1], 0 if parameters is None else len(parameters)
    kernel = _inverse_native.create_transport(n, p)
    if kernel is None:
        pytest.skip("Optional C compiler/native library is unavailable.")
    f, right, left, timing = [np.zeros(n) for _ in range(4)]
    df, dright, dleft, dtiming = [np.zeros((n, p)) for _ in range(4)]
    for ti in range(1, len(t)):
        interval = t[ti] - t[ti - 1]
        substeps = max(1, int(np.ceil(np.max(speed[ti - 1:ti + 1]) * interval / (0.8 * resolution))))
        derivatives = [None] * 4
        if p:
            derivatives = [
                field.derivative(value, index, p)
                for field, value in zip(fields, (firing, speed))
                for index in (ti - 1, ti)
            ]
        kernel.advance(
            interval, substeps, resolution, periodic,
            firing[ti - 1], firing[ti], speed[ti - 1], speed[ti], *derivatives,
            f, right, left, timing, df, dright, dleft, dtiming,
        )
    return timing, dtiming, f


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("n,rate", [(1, 0.02), (13, 0.02), (13, 1.0)])
@pytest.mark.parametrize("forms", list(product(("constant", "space", "time", "spacetime"), repeat=2)))
def test_native_matches_numpy_on_all_form_pairs_and_completion_branches(monkeypatch, periodic, n, rate, forms):
    t = np.array([0.0, 0.17, 1.4, 3.0, 9.0, 18.0, 33.0])
    first = _make_field(forms[0], None, rate, (1e-6, 3), n, t, 2, 3, 2, 0, "firing_rate")
    second = _make_field(forms[1], None, 1.4, (0.2, 6), n, t, 2, 3, 2, first.count, "fork_speed")
    parameters = np.concatenate([first.initial, second.initial])
    parameters += np.linspace(-0.2, 0.3, parameters.size)
    fields = (first, second)
    firing, speed = [field.evaluate(parameters) for field in fields]
    actual = run_native(firing, speed, t, 1.3, periodic, fields, parameters)
    monkeypatch.setenv("DNASCAPE_DISABLE_NATIVE", "1")
    expected = _mean_field_timing(firing, speed, t, 1.3, periodic, fields, parameters)
    for native, numpy in zip(actual, expected):
        np.testing.assert_allclose(native, numpy, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_native_forward_without_parameters_matches_numpy(monkeypatch, periodic):
    t = np.arange(0.0, 60.0, 2.0)
    firing = np.broadcast_to(0.008, (len(t), 17))
    speed = np.broadcast_to(np.linspace(0.8, 2.0, 17), firing.shape)
    actual = run_native(firing, speed, t, 4.0, periodic)
    monkeypatch.setenv("DNASCAPE_DISABLE_NATIVE", "1")
    expected = _mean_field_timing(firing, speed, t, 4.0, periodic)
    for native, numpy in zip(actual, expected):
        np.testing.assert_allclose(native, numpy, rtol=2e-13, atol=2e-13)


def test_native_can_be_disabled_without_loading_a_compiler(monkeypatch):
    monkeypatch.setenv("DNASCAPE_DISABLE_NATIVE", "1")
    monkeypatch.setattr(_inverse_native, "_compiler", lambda: pytest.fail("Compiler should not run."))
    assert _inverse_native.create_transport(10, 2) is None


def test_missing_compiler_falls_back_once(monkeypatch, tmp_path):
    monkeypatch.delenv("DNASCAPE_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_inverse_native, "_ATTEMPTED", False)
    monkeypatch.setattr(_inverse_native, "_LIBRARY", None)
    monkeypatch.setattr(_inverse_native, "_UNAVAILABLE_REASON", None)
    monkeypatch.setattr(_inverse_native.tempfile, "gettempdir", lambda: str(tmp_path))
    calls = []
    monkeypatch.setattr(_inverse_native, "_compiler", lambda: calls.append(1))
    assert _inverse_native.create_transport(10, 2) is None
    assert _inverse_native.create_transport(10, 2) is None
    assert len(calls) == 1
    assert "No supported C compiler" in _inverse_native._UNAVAILABLE_REASON


@pytest.mark.parametrize("error", [
    OSError("Compiler could not start."),
    subprocess.CalledProcessError(1, ["compiler"]),
    subprocess.TimeoutExpired(["compiler"], 15),
])
def test_failed_compilation_falls_back_without_repeated_attempts(monkeypatch, tmp_path, error):
    monkeypatch.delenv("DNASCAPE_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_inverse_native, "_ATTEMPTED", False)
    monkeypatch.setattr(_inverse_native, "_LIBRARY", None)
    monkeypatch.setattr(_inverse_native, "_UNAVAILABLE_REASON", None)
    monkeypatch.setattr(_inverse_native, "_compiler", lambda: "compiler")
    monkeypatch.setattr(_inverse_native.tempfile, "gettempdir", lambda: str(tmp_path))
    calls = []

    def fail_compile(*args, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(_inverse_native.subprocess, "run", fail_compile)
    assert _inverse_native.create_transport(10, 2) is None
    assert _inverse_native.create_transport(10, 2) is None
    assert len(calls) == 1
    assert calls[0]["timeout"] == 15
    assert _inverse_native._UNAVAILABLE_REASON


def test_compiler_receives_the_captured_bytes_used_for_the_cache_key(monkeypatch, tmp_path):
    original_source = Path(_inverse_native.__file__).with_name("_inverse_transport.c")
    expected = original_source.read_bytes()
    monkeypatch.delenv("DNASCAPE_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_inverse_native, "_ATTEMPTED", False)
    monkeypatch.setattr(_inverse_native, "_LIBRARY", None)
    monkeypatch.setattr(_inverse_native, "_UNAVAILABLE_REASON", None)
    monkeypatch.setattr(_inverse_native, "_compiler", lambda: "compiler")
    monkeypatch.setattr(_inverse_native.tempfile, "gettempdir", lambda: str(tmp_path))
    captured = []

    def inspect_compile(command, **kwargs):
        source = Path(command[-3])
        assert source != original_source
        assert source.read_bytes() == expected
        captured.append(source)
        # The source assertion needs no real build or compiler in CI.
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(_inverse_native.subprocess, "run", inspect_compile)
    assert _inverse_native.create_transport(5, 0) is None
    assert len(captured) == 1


def test_cached_library_does_not_require_compiler_to_remain_installed(monkeypatch):
    if _inverse_native.create_transport(5, 0) is None:
        pytest.skip("Optional C compiler/native library is unavailable.")
    monkeypatch.setattr(_inverse_native, "_ATTEMPTED", False)
    monkeypatch.setattr(_inverse_native, "_LIBRARY", None)
    monkeypatch.setattr(_inverse_native, "_compiler", lambda: pytest.fail("Cache must avoid compiler."))
    assert _inverse_native.create_transport(5, 0) is not None


def test_missing_native_symbol_falls_back(monkeypatch):
    if _inverse_native.create_transport(5, 0) is None:
        pytest.skip("Optional C compiler/native library is unavailable.")
    monkeypatch.setattr(_inverse_native, "_ATTEMPTED", False)
    monkeypatch.setattr(_inverse_native, "_LIBRARY", None)
    monkeypatch.setattr(_inverse_native, "_UNAVAILABLE_REASON", None)
    monkeypatch.setattr(_inverse_native.ctypes, "CDLL", lambda path: object())
    assert _inverse_native.create_transport(5, 0) is None
    assert "inverse_transport_abi_version" in _inverse_native._UNAVAILABLE_REASON


def test_mismatched_native_abi_is_rejected_before_resolving_kernel(monkeypatch):
    if _inverse_native.create_transport(5, 0) is None:
        pytest.skip("Optional C compiler/native library is unavailable.")

    class WrongABIVersion:
        def __call__(self):
            return _inverse_native._ABI_VERSION + 1

    class LibraryWithoutKernel:
        inverse_transport_abi_version = WrongABIVersion()

    monkeypatch.setattr(_inverse_native, "_ATTEMPTED", False)
    monkeypatch.setattr(_inverse_native, "_LIBRARY", None)
    monkeypatch.setattr(_inverse_native, "_UNAVAILABLE_REASON", None)
    monkeypatch.setattr(_inverse_native.ctypes, "CDLL", lambda path: LibraryWithoutKernel())
    assert _inverse_native.create_transport(5, 0) is None
    assert "ABI does not match" in _inverse_native._UNAVAILABLE_REASON


@pytest.mark.parametrize("periodic", [False, True])
def test_native_chunking_preserves_global_midpoints_and_single_trapezoid(monkeypatch, periodic):
    t, n = np.array([0.0, 360.0, 381.0]), 13
    firing = _make_field("time", None, 0.00002, (1e-8, 1), n, t, 2, 3, 2, 0, "firing_rate")
    speed = _make_field("spacetime", None, 1.4, (0.2, 6), n, t, 2, 3, 2, firing.count, "fork_speed")
    fields = (firing, speed)
    parameters = np.concatenate([field.initial for field in fields])
    parameters += np.linspace(-0.3, 0.3, parameters.size)
    first, second = [field.evaluate(parameters) for field in fields]
    actual = run_native(first, second, t, 1.0, periodic, fields, parameters)
    monkeypatch.setenv("DNASCAPE_DISABLE_NATIVE", "1")
    expected = _mean_field_timing(first, second, t, 1.0, periodic, fields, parameters)
    for native, numpy in zip(actual, expected):
        np.testing.assert_allclose(native, numpy, rtol=2e-12, atol=2e-12)


def test_native_cancels_inside_a_single_long_time_interval(monkeypatch):
    kernel = _inverse_native.create_transport(13, 0)
    if kernel is None:
        pytest.skip("Optional C compiler/native library is unavailable.")
    checks = []

    def cancel_second_chunk():
        checks.append(1)
        if len(checks) == 2:
            raise RuntimeError("Stop requested inside interval.")

    monkeypatch.setattr(_inverse_native, "check_cancelled", cancel_second_chunk)
    f, right, left, timing = [np.zeros(13) for _ in range(4)]
    derivatives = [np.zeros((13, 0)) for _ in range(4)]
    with pytest.raises(RuntimeError, match="Stop requested"):
        kernel.advance(
            1000.0, 1000, 1.0, True, 0.00001, 0.00001, 0.2, 0.2,
            None, None, None, None, f, right, left, timing, *derivatives,
        )
    assert len(checks) == 2
    assert np.any(f > 0)
    # An interrupted state is discarded by the fit, never integrated as if
    # the entire requested interval had completed.
    assert not np.any(timing)
