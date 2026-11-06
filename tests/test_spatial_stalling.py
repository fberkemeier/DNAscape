"""Explicit spatial stalls, recovery, censoring and native-grid workflows."""
import numpy as np
import pytest

from dnascape.simulation import rsim, simulate_replication, stall_probability_per_bin
from dnascape.workflows.simulation import ConstantSimulationRequest, run_constant_simulation, validate_constant_simulation


def ensemble(**kwargs):
    options = dict(initiation_rate=.02, fork_speed=1.4, n_bins=41, sim_number=9,
                   seed=27, verbose=False, max_rep_time=10_000.,
                   time_statsQ=True, time_grid=np.arange(0., 101., 2.))
    options.update(kwargs)
    return rsim(**options)


def same_observations(left, right):
    for key in ("rep_times_per_sim", "replication_timing", "fork_directionality", "origin_efficiency",
                "completion_time", "s_phase_duration", "replicated_fraction", "total_forks", "stalled_forks"):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("std", [0., .3])
@pytest.mark.parametrize("shared", [False, True])
def test_constant_spatial_profiles_exactly_reproduce_scalar_stalls(periodic, std, shared):
    options = dict(perQ=periodic, fork_speed_std=std, stall_shared_directions=shared)
    scalar = ensemble(stall_rate_per_kb=.12, tau=7., **options)
    spatial = ensemble(source={"stall_rate_per_kb": np.full(41, .12), "tau": np.full(41, 7.)}, **options)
    rate = ensemble(stall_rate_per_kb=.12, stall_recovery_rate=1/7, **options)
    same_observations(scalar, spatial)
    same_observations(scalar, rate)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("std", [0., .3])
def test_zero_stalling_consumes_no_extra_random_draws(periodic, std):
    baseline = ensemble(perQ=periodic, fork_speed_std=std)
    zero = ensemble(source={"stall_rate_per_kb": np.zeros(41), "stall_recovery_rate": np.linspace(0., .5, 41)},
                    perQ=periodic, fork_speed_std=std, stall_shared_directions=True)
    same_observations(baseline, zero)


@pytest.mark.parametrize("periodic", [False, True])
def test_spatial_permanent_barrier_and_finite_recovery(periodic):
    source = dict(initiation_rate=np.r_[1e9, np.zeros(8)], fork_speed=1.,
                  stall_rate_per_kb=np.array([0., 0., 0., 0., 1e6, 0., 0., 0., 0.]),
                  stall_recovery_rate=np.zeros(9))
    options = dict(sim_number=1, seed=3, verbose=False, geometry="periodic" if periodic else "linear",
                   max_rep_time=100., time_grid=np.arange(0., 101.), time_statsQ=True, time_stats_xtQ=True)
    permanent = rsim(source=source, assumptions=options)
    censored = [4] if periodic else [4, 5, 6, 7, 8]
    assert np.isinf(permanent["rep_times_per_sim"][0, censored]).all()
    assert np.isinf(permanent["completion_time"][0])
    assert permanent["incomplete_simulations"] == 1
    assert not permanent["completed"][0]
    expected_fraction = (9 - len(censored)) / 9
    assert permanent["replicated_fraction"][-1] == pytest.approx(expected_fraction)
    assert permanent["replicated_fraction_xt"][:, -1].mean() == pytest.approx(expected_fraction)
    source["stall_recovery_rate"] = np.full(9, .5)
    recovered = rsim(source=source, assumptions=options)
    assert np.isfinite(recovered["completion_time"]).all()
    assert recovered["replicated_fraction"][-1] == 1.
    assert recovered["replication_timing"][4] > 4.


def test_sampled_pause_has_the_local_exponential_distribution():
    result = ensemble(initiation_rate=np.array([1e9, 0.]), fork_speed=1., n_bins=2, sim_number=2500,
                      stall_rate_per_kb=np.array([0., 1e6]), stall_recovery_rate=np.array([1., .5]))
    pause = result["rep_times_per_sim"][:, 1] - result["rep_times_per_sim"][:, 0] - 1.
    assert pause.min() >= 0.
    assert pause.mean() == pytest.approx(2., abs=.12)
    assert pause.var() == pytest.approx(4., abs=.6)


@pytest.mark.parametrize("shared", [False, True])
def test_shared_directions_reuse_the_same_stall_and_recovery_draws(shared):
    class LoggedRng:
        def __init__(self):
            self.rng = np.random.default_rng(7)
            self.random_calls = 0
            self.recovery_calls = 0

        def random(self, size):
            self.random_calls += 1
            return self.rng.random(size)

        def exponential(self, scale, size=None):
            if np.all(np.asarray(scale) == 2.):
                self.recovery_calls += 1
            return self.rng.exponential(scale=scale, size=size)

    rng = LoggedRng()
    simulate_replication(1e6, 1., n_bins=31, stall_rate_per_kb=np.full(31, 1e6),
                         tau=np.full(31, 2.), stall_shared_directions=shared, rng=rng)
    assert rng.random_calls == (1 if shared else 2)
    assert rng.recovery_calls == (1 if shared else 2)


def test_cutoff_censors_future_replication_and_late_initiations():
    result = ensemble(initiation_rate=np.r_[1e9, np.zeros(7)], fork_speed=.1, n_bins=8,
                      sim_number=3, max_rep_time=10., time_grid=np.arange(21.), time_stats_xtQ=True)
    assert np.isinf(result["rep_times_per_sim"][:, 1:]).all()
    assert np.isinf(result["completion_time"]).all()
    assert np.isinf(result["s_phase_duration"]).all()
    np.testing.assert_array_equal(result["replicated_fraction"][10:], np.full(11, .125))
    np.testing.assert_array_equal(result["replication_timing_observed_fraction"], [1., 0., 0., 0., 0., 0., 0., 0.])
    never_started = ensemble(initiation_rate=1e-30, max_rep_time=1.)
    assert np.isnan(never_started["completion_time"]).all()
    assert np.isnan(never_started["s_phase_duration"]).all()
    assert np.asarray(never_started["num_origins"]).sum() == 0
    assert never_started["replicated_fraction"].sum() == 0.


def test_probability_and_spatial_axis_inference():
    density = np.array([0., .2, .4, 1e4])
    np.testing.assert_array_equal(stall_probability_per_bin(density, .5), -np.expm1(-density * .5))
    inferred = rsim(source={"initiation_rate": .1, "stall_rate_per_kb": density}, seed=3, verbose=False)
    assert inferred["replication_timing"].shape == (4,)


@pytest.mark.parametrize("kwargs, message", [
    ({"stall_rate_per_kb": [-1., 1.]}, "non-negative"),
    ({"stall_rate_per_kb": [0., np.nan]}, "non-negative"),
    ({"stall_recovery_rate": [-1., 1.]}, "non-negative"),
    ({"stall_recovery_rate": np.inf}, "non-negative"),
    ({"tau": [0., 1.]}, "positive"),
    ({"tau": [[1., 2.]]}, "spatial"),
    ({"tau": 2., "stall_recovery_rate": .5}, "either"),
    ({"stall_rate": .2, "stall_rate_per_kb": .1}, "cannot be combined"),
    ({"stall_rate_per_kb": np.zeros(5)}, "Spatial lengths|bins|values|spatial"),
])
def test_invalid_and_conflicting_inputs_fail_clearly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rsim(.02, n_bins=2, verbose=False, **kwargs)


def request(**overrides):
    settings = dict(initiation_rate=.05, fork_speed=1.4, resolution_x=2., resolution_t=1.,
                    region_start_kb=2., region_end_kb=10., replicates=3, geometry="linear",
                    time_window=100., max_replication_time=1000., source_resolution_x=1.)
    settings.update(overrides)
    return ConstantSimulationRequest(**settings)


def test_all_spatial_fields_share_cropping_and_resampling():
    base = np.arange(12.)
    validated = validate_constant_simulation(request(initiation_rate=.01 + base / 1000,
        fork_speed=1. + base / 10, stall_rate_per_kb=base / 20, stall_recovery_rate=base / 30)).request
    indices = np.arange(2., 10.).reshape(4, 2).mean(axis=1)
    for field, expected in (("initiation_rate", .01 + indices / 1000), ("fork_speed", 1 + indices / 10),
                            ("stall_rate_per_kb", indices / 20), ("stall_recovery_rate", indices / 30)):
        np.testing.assert_allclose(getattr(validated, field), expected)
        assert not getattr(validated, field).flags.writeable
    result = run_constant_simulation(validated)
    np.testing.assert_array_equal(result.position_kb, [2., 4., 6., 8.])
    np.testing.assert_array_equal(result.outputs["input:stall_recovery_rate"], validated.stall_recovery_rate)


def test_infinite_recovery_times_survive_non_aligned_resampling():
    validated = validate_constant_simulation(request(
        stall_duration=np.array([2., np.inf, 4., 6.]), stall_rate_per_kb=.2,
        region_start_kb=.5, region_end_kb=4., resolution_x=1.5)).request
    assert np.isinf(validated.stall_duration[0])
    assert not np.isnan(validated.stall_duration).any()


def test_workflow_missing_policy_and_recovery_conflicts():
    values = np.array([0., .2, np.nan, .6] * 3)
    validated = validate_constant_simulation(request(stall_rate_per_kb=values, missing_values="interpolate"))
    assert np.isfinite(validated.request.stall_rate_per_kb).all()
    with pytest.raises(ValueError, match="either"):
        validate_constant_simulation(request(stall_duration=10., stall_recovery_rate=.1))
    with pytest.raises(ValueError, match="spatial"):
        validate_constant_simulation(request(stall_duration=np.ones((1, 12))))


def test_spatial_inputs_belong_to_source_and_legacy_scalar_assumptions_still_work():
    same_observations(ensemble(stall_rate_per_kb=.1, tau=10.),
                      ensemble(assumptions={"stall_rate_per_kb": .1, "tau": 10.}))
    with pytest.raises(ValueError, match="source"):
        ensemble(assumptions={"stall_rate_per_kb": np.ones(41)})
    with pytest.raises(ValueError, match="not both"):
        ensemble(tau=10., source={"tau": np.ones(41)})


def test_workflow_reports_incomplete_runs_and_exports_censored_values():
    from dnascape.workflows.exports import build_raw_export
    import io
    result = run_constant_simulation(request(initiation_rate=np.r_[1e9, np.zeros(7)],
        stall_rate_per_kb=np.full(8, 1e6), stall_recovery_rate=np.zeros(8),
        resolution_x=1., region_start_kb=0., region_end_kb=8., track_time_outputs=True))
    assert result.incomplete_simulations == 3
    np.testing.assert_array_equal(result.replication_timing_observed_fraction, [1., 0., 0., 0., 0., 0., 0., 0.])
    artifact = build_raw_export(result, "replication_timing", "npz")
    with np.load(io.BytesIO(artifact.data)) as payload:
        assert np.isinf(payload["values"][1:]).all()


@pytest.mark.parametrize("file_format", ["tsv", "csv", "txt", "npy", "npz", "tsv.gz"])
def test_positive_infinity_is_scoped_to_recovery_time_uploads(file_format):
    import gzip
    import io
    from dnascape.workflows.uploads import inspect_upload, materialize_upload
    values = np.array([10., np.inf, 5., 2.])
    stream = io.BytesIO()
    if file_format == "npy":
        np.save(stream, values)
        data = stream.getvalue()
    elif file_format == "npz":
        np.savez(stream, values=values, data_form="space")
        data = stream.getvalue()
    else:
        data = b"recovery\n10\ninf\n5\n2\n"
        if file_format.endswith(".gz"):
            data = gzip.compress(data)
    name = "recovery." + file_format
    with pytest.raises(ValueError, match="infinit|numeric"):
        inspect_upload(name, data, expected_form="x")
    inspection = inspect_upload(name, data, expected_form="x", allow_positive_infinity=True)
    np.testing.assert_array_equal(materialize_upload(inspection), values)
    # The opt-in is reset even when another upload is read on the same thread.
    with pytest.raises(ValueError, match="infinit|numeric"):
        inspect_upload(name, data, expected_form="x")


def test_infinite_mean_time_and_missing_values_remain_distinct():
    from dnascape.workflows.uploads import inspect_upload, materialize_upload
    inspection = inspect_upload("recovery.tsv", b"tau\n10\n\ninf\n5\n", expected_form="x", allow_positive_infinity=True)
    zero = materialize_upload(inspection, missing_policy="zero")
    assert zero[1] == 0. and np.isinf(zero[2])
    interpolated = materialize_upload(inspection, missing_policy="interpolate")
    assert np.isinf(interpolated[1:3]).all()


def test_infinite_recovery_upload_is_preserved_in_exported_recipe(tmp_path):
    from app.code_export import CodeInputs, CodeSnapshot, build_python
    from dnascape.workflows.uploads import inspect_upload, materialize_upload
    from dnascape.workflows.code_recipes import restore_settings
    import runpy
    inputs = CodeInputs()
    inspection = inspect_upload("tau.tsv", b"tau\ninf\ninf\ninf\ninf\n", expected_form="x", allow_positive_infinity=True)
    encoded = inputs.encode(inspection)
    for name, data in inputs.files.items():
        path = tmp_path / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    restored = restore_settings(encoded, input_directory=tmp_path / "inputs")
    assert restored.allow_positive_infinity
    assert np.isinf(materialize_upload(restored)).all()
    recipe = dict(inputs={
        "initiation_rate": {"mode": "scalar", "form": "x", "values": [.02]},
        "fork_speed": {"mode": "scalar", "form": "x", "values": [1.4]},
        "stall_rate_per_kb": {"mode": "scalar", "form": "x", "values": [.3]},
        "stall_duration": {"mode": "profile", "form": "x", "files": [{"source": encoded}]},
    }, settings=dict(resolution_x=1., resolution_t=1., region_start_kb=0.,
                     replicates=3, geometry="linear", time_window=100., max_replication_time=1000.,
                     seed=3, source_resolution_x=1.), region_end_kb=4., automatic_region_end=4.,
       export={"file_type": "NPZ"})
    artifact = build_python(CodeSnapshot("simulation", recipe, tuple(inputs.files.items())))
    script = tmp_path / artifact.filename
    script.write_bytes(artifact.data)
    namespace = runpy.run_path(str(script))
    result = namespace["main"]()[0][1]
    assert np.isinf(result.outputs["input:stall_duration"]).all()


def test_column_recovery_profile_interpolates_permanent_and_missing_bins():
    import io
    from dnascape.workflows.uploads import inspect_upload, materialize_upload
    stream = io.BytesIO()
    np.savez(stream, values=np.array([[np.inf], [np.nan], [np.inf]]), data_form="space")
    inspection = inspect_upload("tau.npz", stream.getvalue(), expected_form="x", allow_positive_infinity=True)
    assert np.isinf(materialize_upload(inspection, missing_policy="interpolate")).all()


def test_infinity_upload_policy_does_not_leak_to_parallel_reads():
    from concurrent.futures import ThreadPoolExecutor
    from dnascape.workflows.uploads import inspect_upload, materialize_upload
    def read(allow):
        try:
            inspection = inspect_upload("tau.tsv", b"tau\n1\ninf\n3\n", expected_form="x", allow_positive_infinity=allow)
            return np.isinf(materialize_upload(inspection)[1])
        except ValueError:
            return False
    allowed = [True, False] * 12
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, allowed)) == allowed


def test_recipe_automatic_extent_respects_native_profile_spacing():
    from dnascape.workflows.simulation_recipe import prepare_simulation_recipe
    recipe = dict(inputs={
        "initiation_rate": {"mode": "scalar", "form": "x", "values": [.02]},
        "fork_speed": {"mode": "scalar", "form": "x", "values": [1.4]},
    }, settings=dict(resolution_x=1., resolution_t=1., region_start_kb=0.,
                     replicates=3, geometry="linear", time_window=100., max_replication_time=1000.,
                     source_resolution_x=2., stall_rate_per_kb=np.arange(8.) / 10),
       region_end_kb=16., automatic_region_end=16.)
    prepared = prepare_simulation_recipe(recipe)[0][1]
    assert prepared.region_end_kb == 16.
    actual = validate_constant_simulation(prepared).request
    assert actual.stall_rate_per_kb.shape == (16,)
