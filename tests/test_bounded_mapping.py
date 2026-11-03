"""Equivalence to the original exact kernels, including reverse derivatives."""

import numpy as np
import pytest

from dnascape._kinetic_fit import _KineticModel, _trapz_weights, fit_kinetic


def model(form="xt", target="initiation_rate", n=73, clock=None):
    clock = np.arange(171) * 5. if clock is None else clock
    return _KineticModel(target, form, n, clock, 100., np.zeros(1), 2., True,
                         (1e-12, 1.) if target == "initiation_rate" else (.1, 10.))


def dense_objective(m, p, op, observed, valid, rate=None):
    fitted, state = m.evaluate(p)
    rate = fitted if rate is None else rate
    hazard = op.forward(rate, record=True)
    survival = np.exp(-hazard)
    weights = _trapz_weights(m.clock)
    prediction = weights @ survival + 3.
    q = np.where(valid, prediction - observed, 0.) * 2. / (valid.sum() * 100.)
    gi, gv = op.backward(-weights[:, None] * survival * q, with_speed=m.target == "fork_speed")
    gradient = m.gradient(gi if m.target == "initiation_rate" else gv, p, state)
    return prediction, survival[-1], gradient


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("radius", [18., 2000.])
def test_block_cone_matches_original_forward_and_adjoint(periodic, radius, monkeypatch):
    from dnascape._hazard_fft import ConstantSpeedConvolution
    from dnascape._kinetic_bounded import ConeConvolutionTiming

    monkeypatch.delenv("RENDER", raising=False)
    m = model()
    # Independent shapes and delays exercise a non-separable XT field.
    m.delay = np.linspace(0., 12., m.n)
    p = m.initial(None, np.linspace(25., 70., m.n), 1.4)
    p[m.shape_slice] = np.linspace(np.log(.8), np.log(2.4), m.n)
    observed, valid = np.linspace(30., 90., m.n), np.ones(m.n, dtype=bool)
    valid[::7] = False
    speed = np.broadcast_to(1.4, (len(m.clock), m.n))
    op = ConstantSpeedConvolution(speed, m.clock, periodic=periodic, radius=radius)
    expected, final, gradient = dense_objective(m, p, op, observed, valid)
    bounded = ConeConvolutionTiming(m, observed, valid, 3., 100., 1., periodic, radius, speed=1.4, dt=5.)
    actual = bounded.evaluate(p, True)
    np.testing.assert_allclose(actual["prediction"], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["survival"][-1], final, atol=1e-12)
    np.testing.assert_allclose(actual["gradient"], gradient, rtol=1e-8, atol=1e-10)
    deferred = bounded.evaluate(p, False)
    bounded.complete(deferred)
    np.testing.assert_allclose(deferred["gradient"], gradient, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("target", ["initiation_rate", "fork_speed"])
@pytest.mark.parametrize("periodic", [False, True])
def test_block_uniform_matches_original_forward_and_adjoint(target, periodic, monkeypatch):
    from dnascape._hazard_uniform import UniformKineticsHazard
    from dnascape._kinetic_bounded import UniformTiming

    monkeypatch.delenv("RENDER", raising=False)
    m = model("t", target, n=511)
    fixed = 1.4 if target == "initiation_rate" else .002
    p = m.initial(None, np.linspace(25., 70., m.n), fixed)
    fitted, _ = m.evaluate(p)
    rate, speed = (fitted, np.broadcast_to(fixed, fitted.shape)) if target == "initiation_rate" else (np.broadcast_to(fixed, fitted.shape), fitted)
    observed, valid = np.linspace(30., 90., m.n), np.ones(m.n, dtype=bool)
    valid[::7] = False
    op = UniformKineticsHazard(speed, m.clock, periodic=periodic, radius=2000.)
    expected, final, gradient = dense_objective(m, p, op, observed, valid, rate)
    bounded = UniformTiming(m, observed, valid, 3., 100., 1., periodic, 2000.,
                             supplied=fixed, native_clock=m.clock, dt=5.)
    actual = bounded.evaluate(p, True)
    np.testing.assert_allclose(actual["prediction"], expected, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(actual["survival"][-1], final, atol=1e-12)
    np.testing.assert_allclose(actual["gradient"], gradient, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("target", ["initiation_rate", "fork_speed"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_stationary_blocks_match_original_with_temporal_speed(target, periodic, cache, monkeypatch):
    from dnascape import _resources as r
    from dnascape._hazard_stationary_source import StationarySourceHazard
    from dnascape._kinetic_bounded import StationaryConvolutionTiming

    monkeypatch.delenv("RENDER", raising=False)
    m = model("x" if target == "initiation_rate" else "t", target)
    fixed = 1.4 if target == "initiation_rate" else .002 * (1 + .2 * np.sin(np.arange(m.n)))
    p = m.initial(None, np.linspace(25., 70., m.n), fixed)
    fitted, _ = m.evaluate(p)
    speed = np.broadcast_to((1. + .001 * m.clock)[:, None], fitted.shape) if target == "initiation_rate" else fitted
    rate = fitted if target == "initiation_rate" else np.broadcast_to(fixed, fitted.shape)
    observed, valid = np.linspace(30., 90., m.n), np.ones(m.n, dtype=bool)
    valid[::7] = False
    op = StationarySourceHazard(speed, m.clock, periodic=periodic, radius=18.)
    expected, final, gradient = dense_objective(m, p, op, observed, valid, rate)
    if not cache:
        monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 427*r.MIB - 100*1024, 85*r.MIB))
        import dnascape._kinetic_bounded as b
        monkeypatch.setattr(b, "memory_status", r.memory_status)
    bounded = StationaryConvolutionTiming(m, observed, valid, 3., 100., 1., periodic, 18.,
                                          speed=speed if target == "initiation_rate" else None,
                                          rate=rate if target == "fork_speed" else None)
    actual = bounded.evaluate(p, True)
    assert bounded.cache_kernel == cache
    np.testing.assert_allclose(actual["prediction"], expected, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(actual["survival"][-1], final, atol=1e-12)
    np.testing.assert_allclose(actual["gradient"], gradient, rtol=1e-10, atol=1e-10)


def test_large_xt_uses_finite_history_and_bounds_model_batches(monkeypatch):
    import dnascape._kinetic_bounded as b
    from dnascape import _resources as r

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    monkeypatch.setattr(b, "memory_status", r.memory_status)
    m = model(n=10000, clock=np.arange(1201) * 5.)
    observed, valid = np.linspace(30., 500., m.n), np.ones(m.n, dtype=bool)
    bounded = b.ConeConvolutionTiming(m, observed, valid, 0., 100., 1., False, 2000., speed=1.4, dt=5.)
    assert bounded.shape[0] < len(m.clock) // 2
    p = m.initial(None, observed, 1.4)
    original = _KineticModel.evaluate
    sizes = []

    def small_only(self, parameters, **kwargs):
        sizes.append(len(self.clock))
        assert len(self.clock) <= 32
        return original(self, parameters, **kwargs)

    monkeypatch.setattr(_KineticModel, "evaluate", small_only)
    history = bounded.rate_window(p, 300)
    next_history = bounded.rate_window(p, 300 + bounded.block)
    assert next_history is history
    assert max(sizes) <= 32


def test_huge_cone_is_declined_before_transform_allocation(monkeypatch):
    import dnascape._kinetic_bounded as b
    from dnascape import _resources as r

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    monkeypatch.setattr(b, "memory_status", r.memory_status)
    m = model(n=100000, clock=np.arange(1201) * 5.)
    monkeypatch.setattr(b, "rfftn", lambda *a, **k: pytest.fail("transform allocated before admission"))
    with pytest.raises(r.HostedResourceError, match="Online memory limit"):
        b.ConeConvolutionTiming(m, np.ones(m.n), np.ones(m.n, dtype=bool), 0., 100., 1., False, 2000., speed=1.4, dt=5.)


@pytest.mark.parametrize("form", ["space", "time", "spacetime"])
def test_hosted_matches_local_fitting_history(form, monkeypatch):
    from dnascape import _resources as r

    options = dict(target="initiation_rate", target_form=form, resolution=1.,
                   resolution_time=5., maxiter=3, tolerance=0., adaptive_fitting=False)
    observed = np.linspace(70., 100., 80)
    monkeypatch.delenv("RENDER", raising=False)
    local = fit_kinetic(observed, 1.4, **options)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    online = fit_kinetic(observed, 1.4, **options)
    np.testing.assert_allclose(online, local, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(online._rmap_info["fit_history"], local._rmap_info["fit_history"], rtol=1e-9)
    assert "blocked" in online._rmap_info["forward_backend"]


@pytest.mark.parametrize("form", ["time", "spacetime"])
@pytest.mark.parametrize("clock", [np.arange(151) * 2., np.array([0., 3., 10., 30., 80., 160., 300.])])
def test_custom_time_grid_keeps_original_numerics(form, clock, monkeypatch):
    from dnascape import _resources as r

    options = dict(target="initiation_rate", target_form=form, resolution=1.,
                   resolution_time=5., time_grid=clock, maxiter=1,
                   initial_initiation_rate=.01, adaptive_fitting=False)
    observed = np.linspace(20., 30., 19)
    monkeypatch.delenv("RENDER", raising=False)
    local = fit_kinetic(observed, 1.4, **options)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    online = fit_kinetic(observed, 1.4, **options)
    np.testing.assert_allclose(online, local, rtol=1e-7, atol=1e-12)
    np.testing.assert_array_equal(online._rmap_info["time_grid"], local._rmap_info["time_grid"])


@pytest.mark.parametrize("form", ["space", "time", "spacetime"])
def test_hosted_app_timing_routes_complete_with_selected_region(form, monkeypatch):
    from dnascape import _resources as r
    from dnascape.workflows.mapping import MappingRequest, run_mapping
    from dnascape.workflows.mapping_catalog import mapping_route

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    method = {"space": "timing_and_speed_to_initiation_rate",
              "time": "timing_and_speed_to_time_initiation_rate",
              "spacetime": "timing_and_speed_to_spacetime_initiation_rate"}[form]
    settings = {"maxiter": 1} if form == "space" else {"temporal_maxiter": 1}
    if form == "spacetime":
        settings["space_maxiter"] = 1
    messages = []
    observed = np.full(40080, 90.)
    observed[-80:] = np.linspace(70., 100., 80)
    result = run_mapping(MappingRequest(
        mapping_route(method), {"replication_timing": observed, "fork_speed": 1.4},
        {"replication_timing": "x", "fork_speed": "scalar"}, settings=settings,
        region_start_kb=40000., region_end_kb=40080., resolution_x=1., resolution_t=5.,
    ), report=lambda _, message: messages.append(message))
    assert np.isfinite(result.outputs[result.primary_output]).all()
    np.testing.assert_array_equal(result.x_axis, 40000. + np.arange(80))
    assert messages[-1] == "Mapping complete"


@pytest.mark.parametrize("form", ["time", "spacetime"])
def test_hosted_still_rejects_nonpositive_fork_speed(form, monkeypatch):
    from dnascape import _resources as r

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    with pytest.raises(ValueError, match="positive"):
        fit_kinetic(np.linspace(20., 30., 19), 0., target="initiation_rate", target_form=form,
                    maxiter=0, adaptive_fitting=False)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("form", ["t", "xt"])
@pytest.mark.parametrize("sparse", [False, True])
def test_spatial_cone_tiles_keep_one_objective_and_exact_adjoint(periodic, form, sparse, monkeypatch):
    from dnascape._hazard_fft import ConstantSpeedConvolution
    from dnascape._kinetic_bounded import SpatialConeTiming

    monkeypatch.delenv("RENDER", raising=False)
    m = model(form)
    m.delay = np.linspace(0., 12., m.n)
    p = m.initial(None, np.linspace(25., 70., m.n), 1.4)
    observed, valid = np.linspace(30., 90., m.n), np.ones(m.n, dtype=bool)
    valid[::7] = False
    if sparse:
        valid[10:] = False  # Includes tiles with no observed receivers.
    speed = np.broadcast_to(1.4, (len(m.clock), m.n))
    op = ConstantSpeedConvolution(speed, m.clock, periodic=periodic, radius=18.)
    expected, final, gradient = dense_objective(m, p, op, observed, valid)
    bounded = SpatialConeTiming(m, observed, valid, 3., 100., 1., periodic, 18., speed=1.4, dt=5.)
    bounded.core = 11  # Exercise real overlaps, circular seams and edge tiles.
    actual = bounded.evaluate(p, True)
    np.testing.assert_allclose(actual["prediction"], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["survival"][-1], final, atol=1e-12)
    np.testing.assert_allclose(actual["gradient"], gradient, rtol=1e-8, atol=1e-10)
    deferred = bounded.evaluate(p, False)
    bounded.complete(deferred)
    np.testing.assert_allclose(deferred["gradient"], gradient, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(bounded.public(p), m.public(m.evaluate(p)[0]), atol=1e-12)
    assert bounded.cached is None


def test_spatial_fallback_admits_native_grid_with_less_headroom(monkeypatch):
    import dnascape._kinetic_bounded as b
    from dnascape import _resources as r

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 300*r.MIB, 85*r.MIB))
    monkeypatch.setattr(b, "memory_status", r.memory_status)
    m = model(n=10000, clock=np.arange(351) * 5.)
    args = (m, np.linspace(30., 500., m.n), np.ones(m.n, dtype=bool), 0., 100., 1., False, 2000.)
    with pytest.raises(r.HostedResourceError):
        b.ConeConvolutionTiming(*args, speed=1.4, dt=5.)
    bounded = b.SpatialConeTiming(*args, speed=1.4, dt=5.)
    assert 512 <= bounded.core < m.n - 2 * bounded.halo
    assert len(list(bounded.tiles())) > 1


@pytest.mark.parametrize("form,adjoints", [("space", 1), ("time", 1), ("spacetime", 2)])
def test_final_accepted_hosted_proposal_needs_no_unused_adjoint(form, adjoints, monkeypatch):
    import dnascape._kinetic_bounded as b
    from dnascape import _resources as r

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(r, "memory_status", lambda: r.MemoryStatus(512*r.MIB, 160*r.MIB, 85*r.MIB))
    calls = []
    original = b.BoundedTimingFit.complete

    def complete(self, result):
        calls.append(True)
        return original(self, result)

    monkeypatch.setattr(b.BoundedTimingFit, "complete", complete)
    result = fit_kinetic(np.linspace(70., 100., 80), 1.4, target="initiation_rate", target_form=form,
                         maxiter=1, tolerance=0., adaptive_fitting=False)
    assert result._rmap_info["phase_history"][-1]["accepted"]
    assert len(calls) == adjoints
