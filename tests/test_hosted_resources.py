"""Hosted safety boundaries and equivalence of the bounded-memory solver."""

import io
import os
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from dnascape import _resources as resources


@pytest.fixture
def render(monkeypatch, tmp_path):
    monkeypatch.setenv("RENDER", "true")
    limit, usage = tmp_path / "limit", tmp_path / "usage"
    limit.write_text(str(512 * resources.MIB))
    usage.write_text(str(160 * resources.MIB))
    monkeypatch.setattr(resources, "_cgroup_pairs", lambda: [(str(limit), str(usage))])
    monkeypatch.delenv("DNASCAPE_MEMORY_LIMIT_MB", raising=False)
    return usage


def test_detects_container_pressure_and_override_cannot_raise_limit(render, monkeypatch):
    assert resources.memory_status().limit == 512 * resources.MIB
    monkeypatch.setenv("DNASCAPE_MEMORY_LIMIT_MB", "4096")
    assert resources.memory_status().limit == 512 * resources.MIB
    monkeypatch.setenv("DNASCAPE_MEMORY_LIMIT_MB", "384")
    assert resources.memory_status().limit == 384 * resources.MIB
    with pytest.raises(resources.HostedResourceError, match="Online memory limit"):
        resources.ensure_capacity(200 * resources.MIB)
    render.write_text(str(380 * resources.MIB))
    with pytest.raises(resources.HostedResourceError):
        resources.ensure_capacity()


def test_local_has_no_hosted_budget_or_extra_upload_limit(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("DNASCAPE_MEMORY_LIMIT_MB", "1")
    assert resources.memory_status() is None
    assert resources.upload_limit() is None
    resources.ensure_capacity(10**15)


def test_runtime_checkpoint_stops_when_other_sessions_consume_headroom(render, monkeypatch):
    monkeypatch.setattr(resources._STATE, "checked", 0., raising=False)
    render.write_text(str(490 * resources.MIB))
    with pytest.raises(resources.HostedResourceError, match="Online memory limit"):
        from dnascape.utils import check_cancelled
        check_cancelled()


@pytest.mark.skipif(os.name != "posix", reason="Linux container process locking")
def test_heavy_jobs_are_exclusive_across_processes(render):
    code = """
from dnascape import _resources as r
r.memory_status = lambda: r.MemoryStatus(512*1024**2, 100*1024**2, 80*1024**2)
try:
    with r.operation('Simulation'):
        raise SystemExit('unexpected admission')
except r.HostedResourceError as e:
    print(e)
"""
    with resources.operation("Mapping"):
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0
    assert "Online app busy" in completed.stdout


def test_shared_job_guard_releases_on_error_and_allows_nested_calls(render):
    entered, finish = Event(), Event()

    def worker():
        with resources.operation("Simulation"):
            with resources.operation("Nested preparation"):
                entered.set()
                finish.wait(5)

    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(worker)
        assert entered.wait(5)
        try:
            with pytest.raises(resources.HostedResourceError, match="Online app busy"):
                with resources.operation("Mapping"):
                    pass
        finally:
            finish.set()
        running.result()
    with pytest.raises(resources.HostedResourceError):
        with resources.operation("Mapping"):
            raise MemoryError()
    with resources.operation("Data Manager"):
        pass


def test_numpy_header_rejected_before_loader_or_large_allocation(render, monkeypatch):
    from dnascape.workflows.uploads import inspect_upload
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "shape": (10**9,), "fortran_order": False, "descr": "<f8",
    })
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("np.load must not be reached"))
    with pytest.raises(resources.HostedResourceError):
        inspect_upload("large.npy", stream.getvalue())


def test_simulation_declined_before_kinetic_inputs_are_materialized(render, monkeypatch):
    from dnascape.workflows import simulation
    request = simulation.ConstantSimulationRequest(
        initiation_rate=.001, fork_speed=1.4, resolution_x=1., resolution_t=1.,
        region_start_kb=0., region_end_kb=10000., replicates=100,
        geometry="linear", time_window=1000., max_replication_time=1500.,
        track_detailed_outputs=True,
    )
    monkeypatch.setattr(simulation, "_kinetic_field", lambda *a, **k: pytest.fail("allocated too early"))
    with pytest.raises(resources.HostedResourceError):
        simulation.validate_constant_simulation(request)


def test_data_manager_declines_large_expansion_even_with_raised_conversion_setting(render, monkeypatch):
    from dnascape.workflows import conversion, uploads
    inspection = uploads.inspect_upload("rt.tsv", b"value\n30\n40\n")
    inspection = replace(inspection, shape=(10_000_000,))
    request = conversion.DataConversionRequest(inspection=inspection, input_form="x", memory_limit_mb=4096)
    monkeypatch.setattr(conversion, "extract_values", lambda *a, **k: pytest.fail("allocated too early"))
    with pytest.raises(resources.HostedResourceError):
        conversion.prepare_data(request)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("with_speed", [False, True])
def test_streamed_timing_and_gradient_match_full_field(periodic, with_speed, monkeypatch):
    from dnascape._hazard_static import StationaryHazard
    from dnascape._kinetic_fit import _trapz_weights
    from dnascape._stationary_timing import stationary_timing
    monkeypatch.delenv("RENDER", raising=False)
    n = 73
    rate = .003 * (1 + .3 * np.sin(np.arange(n)))
    speed = 1.4 + .2 * np.cos(np.arange(n))
    clock = np.r_[0., np.geomspace(.1, 100., 151)]
    weights = _trapz_weights(clock)
    observed = np.linspace(15., 45., n)
    valid = np.ones(n, dtype=bool)
    valid[::7] = False
    operator = StationaryHazard(speed, clock, 1., periodic, 18.)
    hazard = operator.forward(rate, record=True)
    survival = np.exp(-hazard)
    expected = weights @ survival + 3.
    residual = expected - observed
    adjoint = -weights[:, None] * survival * np.where(valid, residual, 0.)[None, :] * (2 / (valid.sum() * 100))
    gi, gv = operator.backward(adjoint, with_speed=with_speed)
    prediction, error, sensitivity, final, gradient = stationary_timing(
        rate, speed, clock, weights, 1., periodic, 18., observed, valid, 3., 100., with_speed,
    )
    np.testing.assert_allclose(prediction, expected, rtol=1e-12)
    np.testing.assert_allclose(error, residual, atol=1e-12)
    np.testing.assert_allclose(gradient[0], gv if with_speed else gi, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(sensitivity, np.einsum("t,tx,tx->x", weights, hazard, survival), rtol=1e-12)
    np.testing.assert_allclose(final, survival[-1])


def test_rate_only_fft_kernel_preserves_prediction_and_adjoint(monkeypatch):
    from dnascape._hazard_stationary_source import StationarySourceHazard
    monkeypatch.delenv("RENDER", raising=False)
    t = np.linspace(0., 100., 101)
    speed = np.broadcast_to((1. + .01 * t)[:, None], (len(t), 63))
    rate = np.broadcast_to(np.linspace(.001, .003, 63)[None, :], speed.shape)
    full = StationarySourceHazard(speed, t, radius=20.)
    lean = StationarySourceHazard(speed, t, radius=20., differentiate_speed=False)
    np.testing.assert_allclose(lean.forward(rate, True), full.forward(rate, True), atol=1e-12)
    gradient = np.random.default_rng(5).normal(size=rate.shape)
    np.testing.assert_allclose(lean.backward(gradient, with_speed=False)[0],
                               full.backward(gradient, with_speed=False)[0], atol=1e-10)
    assert lean._kernel_state is None


def test_hosted_fit_preserves_native_grid_and_matches_local_fit(render, monkeypatch):
    from dnascape._kinetic_fit import fit_kinetic
    observed = np.linspace(70., 100., 80)
    options = dict(target="initiation_rate", target_form="space", resolution=1.,
                   resolution_time=5., maxiter=3, tolerance=0., adaptive_fitting=False)
    online = fit_kinetic(observed, 1.4, **options)
    assert "streamed" in online._rmap_info["forward_backend"]
    monkeypatch.delenv("RENDER", raising=False)
    local = fit_kinetic(observed, 1.4, **options)
    np.testing.assert_allclose(online, local, rtol=1e-7, atol=1e-12)
    assert online.shape == (80,)


def test_warning_links_to_local_instructions_only_on_render(render, monkeypatch):
    from app import notifications
    calls = []
    monkeypatch.setattr(notifications.ui, "notify", lambda *a, **k: calls.append((a, k)))
    notifications.notify(resources.resource_message("Mapping"), type="negative")
    assert calls[-1][1]["type"] == "warning"
    assert calls[-1][1]["timeout"] == 0
    assert calls[-1][1]["actions"][0]["label"] == "Run locally"
    destinations = []
    monkeypatch.setattr(notifications.ui.navigate, "to", lambda url, **k: destinations.append(url))
    calls[-1][1]["actions"][0]["handler"]()
    assert destinations == [resources.LOCAL_SETUP_URL]
    monkeypatch.delenv("RENDER", raising=False)
    notifications.notify("Existing local limit", type="negative")
    assert calls[-1][1]["type"] == "negative"
    assert all(a.get("label") != "Run locally" for a in calls[-1][1]["actions"])


def test_hosted_cache_budget_is_shared_between_visitors(render, monkeypatch):
    from app import session_cache
    caches = OrderedDict((key, session_cache.SessionCache()) for key in ("first", "second"))
    monkeypatch.setattr(session_cache, "_SESSION_CACHES", caches)
    for cache in caches.values():
        cache.add_reference(name="Result", source="Mapping", detail="", payload=None,
                            navigate_to="mapping", size_bytes=20 * resources.MIB)
    assert sum(cache.size() for cache in caches.values()) <= 32 * resources.MIB
    assert not caches["first"].count()
    assert caches["second"].count() == 1


def test_formula_broadcast_is_declined_before_quadratic_intermediate(render):
    from dnascape.workflows.formulas import evaluate_formula
    with pytest.raises(resources.HostedResourceError):
        evaluate_formula("source[:, None] * source[None, :]", {"source": np.ones(10000)})


@pytest.mark.parametrize("target", ["initiation_rate", "fork_speed"])
@pytest.mark.parametrize("form", ["constant", "space", "time", "spacetime"])
def test_other_hosted_mapping_dependences_remain_available(render, target, form):
    from dnascape._kinetic_fit import fit_kinetic
    result = fit_kinetic(np.linspace(20., 30., 19), 1.4 if target == "initiation_rate" else .002,
                         target=target, target_form=form, resolution=1., maxiter=1,
                         resolution_time=5., adaptive_fitting=False)
    assert np.isfinite(result).all()
    assert result._rmap_info["native_resolution"] == 1.


def test_small_hosted_conversion_simulation_and_download_still_work(render):
    from dnascape.workflows.conversion import DataConversionRequest, convert_data
    from dnascape.workflows.exports import build_raw_export
    from dnascape.workflows.simulation import ConstantSimulationRequest, run_constant_simulation
    from dnascape.workflows.uploads import inspect_upload
    upload = inspect_upload("rt.tsv", b"value\n30\n40\n")
    converted = convert_data(DataConversionRequest(inspection=upload, input_form="x", target_format="npz"))
    assert converted.artifact.data
    result = run_constant_simulation(ConstantSimulationRequest(
        initiation_rate=.1, fork_speed=1.4, resolution_x=1., resolution_t=1.,
        region_start_kb=0., region_end_kb=20., replicates=2, geometry="linear",
        time_window=30., max_replication_time=100., seed=7,
    ))
    assert result.replication_timing.shape == (20,)
    assert build_raw_export(result, "replication_timing", "npz").data
