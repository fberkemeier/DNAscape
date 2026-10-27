"""Downloaded developer scripts reproduce data with no application dependency."""
from __future__ import annotations

import io
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.code_export import CodeInputs, CodeSnapshot, build_python
from dnascape.workflows.code_recipes import restore_settings
from dnascape.workflows.conversion import DataConversionRequest, convert_combined_data, convert_data
from dnascape.workflows.filtering import DataFilterSettings, FilterProfile
from dnascape.workflows.modulation import ProfileModulationSettings
from dnascape.workflows.uploads import inspect_upload


def write_script(snapshot, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in snapshot.files:
        path = folder / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    artifact = build_python(snapshot)
    assert artifact.filename.endswith(".py")
    assert "from app" not in artifact.data.decode()
    path = folder / artifact.filename
    path.write_bytes(artifact.data)
    return path


def _inspection(name, values, form="x"):
    data = io.BytesIO()
    np.savez(data, values=values, data_form={"x": "space", "t": "time", "xt": "spacetime"}[form])
    return inspect_upload(name, data.getvalue(), expected_form=form)


def test_codec_retains_array_settings_even_when_dataclass_comparison_ignores_them(tmp_path):
    inputs = CodeInputs()
    upload = _inspection("modulation.npz", np.arange(8.))
    settings = DataFilterSettings(enabled=True, selection="profile",
        profile=FilterProfile(np.array([1000., 2000.]), layout="positions"),
        action="modulate", modulation=ProfileModulationSettings(values=np.arange(8.), upload=upload))
    encoded = inputs.encode(settings)
    for name, data in inputs.files.items():
        (tmp_path / name).write_bytes(data)
    actual = restore_settings(encoded, input_directory=tmp_path)
    np.testing.assert_array_equal(actual.profile.values, settings.profile.values)
    np.testing.assert_array_equal(actual.modulation.values, settings.modulation.values)
    assert actual.modulation.upload.content == upload.content
    for values in (np.array([np.nan, np.inf, -np.inf]), np.arange(1000.), np.array(["chr1", "chr2"])):
        np.testing.assert_array_equal(restore_settings(inputs.encode(values)), values)


def test_input_names_are_safe_and_case_insensitive_without_losing_different_data():
    inputs = CodeInputs()
    first = inspect_upload("../Signal.tsv", b"value\n1\n2\n")
    second = inspect_upload("signal.tsv", b"value\n3\n4\n")
    assert inputs.upload(first) == "Signal.tsv"
    assert inputs.upload(first) == "Signal.tsv"
    assert inputs.upload(second) == "source_2/signal.tsv"


def test_source_names_cannot_add_statements_to_generated_script_comments():
    import ast
    artifact = build_python(CodeSnapshot("data_manager", {"sources": []},
        (("source.tsv\nraise RuntimeError('filename injection')", b"data"),)))
    parsed = ast.parse(artifact.data.decode("utf-8"))
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(parsed))


def test_shared_source_is_inspected_once_per_recipe_and_reread_on_the_next_run(monkeypatch, tmp_path):
    import dnascape.workflows.uploads as uploads
    source = inspect_upload("signals.csv", b"first,second\n1,2\n3,4\n")
    inputs = CodeInputs()
    encoded = inputs.encode(source)
    (tmp_path / "signals.csv").write_bytes(source.content)
    calls = []
    original = uploads.inspect_upload
    def record(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(uploads, "inspect_upload", record)
    restored = restore_settings([encoded, encoded], input_directory=tmp_path)
    assert restored[0] is restored[1]
    assert len(calls) == 1
    (tmp_path / "signals.csv").write_bytes(b"first,second\n5,6\n7,8\n")
    changed = restore_settings(encoded, input_directory=tmp_path)
    assert len(calls) == 2 and changed.content != source.content


def test_track_format_hint_survives_a_link_filename_without_a_track_extension(tmp_path):
    source = inspect_upload("download.dat", b"chr1\t0\t1000\t2\nchr1\t1000\t2000\t4\n",
                            source_format="bedgraph")
    inputs = CodeInputs()
    request = DataConversionRequest(source, "track", chromosome="chr1", start=0, end=2000,
                                    bin_size=1000, target_format="npz")
    expected = convert_data(request)
    script = write_script(CodeSnapshot("data_manager", dict(sources=[inputs.encode(request)]),
                                      tuple(inputs.files.items())), tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    with np.load(tmp_path / "outputs" / "rdata.npz") as data:
        np.testing.assert_array_equal(data["values"].reshape(-1), expected.preview.values.reshape(-1))


def test_plotting_recipe_preserves_sampling_budget_theme_and_report_names(tmp_path):
    from dnascape.workflows.export_settings import get_export_settings
    inputs = CodeInputs()
    source = _inspection("profile.npz", 2. + np.sin(np.arange(800.) / 30.))
    arguments = dict(plot_type="Line", title="Profile", x_label="Position", y_label="Signal", color="#2596be",
        palette="viridis", histogram_bins=10, density=False, show_legend=True, show_colorbar=False,
        log_y=False, invert_y=False, y_limits=None, series_label="Profile", interactive=True)
    recipe = dict(profiles=[dict(id=[1, "values"], name="Profile", form="x", source=inputs.encode(source), array_key="values")],
        methods=["pearson"], report_names={"1:values": "Edited profile"}, preview_points=100,
        plots=[dict(profiles=[[1, "values"]], arguments=arguments,
            curves=[dict(name="Edited profile", color="#2596be")], presentation=dict(theme_color="#008080"))])
    script = write_script(CodeSnapshot("plotting", recipe, tuple(inputs.files.items())), tmp_path)
    previous = get_export_settings().preview_points
    runpy.run_path(str(script), run_name="__main__")
    assert get_export_settings().preview_points == previous
    html = (tmp_path / "outputs" / "plot_1.html").read_text(encoding="utf-8")
    options = json.loads(html.split("const specification = ", 1)[1].split(";\n", 1)[0])
    assert len(options["series"][0]["data"]) <= 100
    assert options["tooltip"]["axisPointer"]["label"]["backgroundColor"] == "#008080"
    assert "Edited profile" in (tmp_path / "outputs" / "analysis_report.tsv").read_text()


def test_external_selection_and_modulation_survive_conversion_and_filter_export(tmp_path):
    inputs = CodeInputs()
    source = _inspection("source.npz", np.arange(8.) + 1)
    positions = inspect_upload("positions.tsv", b"2000\n4000\n", expected_form="regions")
    modulation = _inspection("modulation.npz", np.array([2., 3., 4., 5.]))
    edit = DataFilterSettings(enabled=True, selection="profile", upload=positions,
        profile_layout="positions", profile_units="bp", action="modulate",
        modulation=ProfileModulationSettings(operation="expression", expression="source * profile",
                                              upload=modulation, align_from_start=True))
    request = DataConversionRequest(source, "x", target_format="npz", output_name="signal",
        profile_binning=True, bin_size=2000, source_bin_size=1000, data_filter=edit,
        transform_formula="source + 1")
    expected = convert_data(request)
    recipe = dict(sources=[inputs.encode(request)], exports=[dict(filter=dict(
        output_name="filter", target_format="npz", layout="intervals"))],
        statistics=dict(filename="statistics", file_format="tsv"))
    script = write_script(CodeSnapshot("data_manager", recipe, tuple(inputs.files.items())), tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    with np.load(tmp_path / "outputs" / "signal.npz") as data:
        np.testing.assert_array_equal(data["values"].reshape(-1), expected.preview.values.reshape(-1))
    with np.load(tmp_path / "outputs" / "filter.npz") as data:
        np.testing.assert_array_equal(data["intervals"], [[2000, 6000]])
    assert (tmp_path / "outputs" / "statistics.tsv").is_file()


@pytest.mark.parametrize("mode,count", [("pairwise", 2), ("all", 4)])
def test_simulation_script_preserves_constant_batch_combinations(tmp_path, mode, count):
    recipe = dict(inputs={
        "initiation_rate": dict(mode="constants", form="scalar", values=[.1, .2]),
        "fork_speed": dict(mode="constants", form="scalar", values=[1., 2.]),
    }, settings=dict(resolution_x=1., source_resolution_x=1., resolution_t=5., region_start_kb=0.,
        replicates=2, geometry="linear", time_window=30., max_replication_time=60., seed=7),
        region_end_kb=8., automatic_region_end=8., combination_mode=mode,
        export=dict(file_type="NPZ", filename="custom"))
    script = write_script(CodeSnapshot("simulation", recipe), tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    assert len(tuple((tmp_path / "outputs").glob("*.npz"))) == count


@pytest.mark.parametrize("file_format", ["tsv", "txt", "csv", "npz"])
@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_data_manager_script_reproduces_configure_transform_filter_and_format(tmp_path, file_format, combine, filtered):
    inputs = CodeInputs()
    edit = DataFilterSettings(enabled=filtered, selection="threshold", comparison="above", threshold=8.,
                              action="replace", value=7.)
    requests = tuple(DataConversionRequest(_inspection(f"source{index}.npz", np.arange(12.) + index), "x",
        output_name=f"output{index}", profile_binning=True, source_bin_size=1000, bin_size=2000,
        start=2000, end=10000, scaling_mode="multiplier", scale=2., transform_formula="source + 1",
        data_filter=edit, target_format=file_format) for index in (1, 2))
    expected = ([convert_combined_data(requests, "source1 + source2", output_name="combined")]
                if combine else [convert_data(request) for request in requests])
    snapshot = CodeSnapshot("data_manager", dict(
        sources=[inputs.encode(request) for request in requests],
        combined_formula="source1 + source2" if combine else None, combined_name="combined",
        exports=[dict(target_format=file_format)] * len(expected),
    ), tuple(inputs.files.items()))
    runpy.run_path(str(write_script(snapshot, tmp_path)), run_name="__main__")
    for result in expected:
        assert (tmp_path / "outputs" / result.artifact.filename).read_bytes() == result.artifact.data


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
@pytest.mark.parametrize("periodic", [False, True])
def test_simulation_script_matches_seeded_workflow_with_supplemental_outputs(tmp_path, form, periodic):
    from dnascape.workflows.simulation import run_constant_simulation
    from dnascape.workflows.simulation_recipe import prepare_simulation_recipe

    inputs = CodeInputs()
    values = {"x": np.full(8, .2), "t": np.full((1, 7), .2), "xt": np.full((8, 7), .2)}
    source = (dict(mode="scalar", form="scalar", values=[.2]) if form == "scalar" else
              dict(mode="profile", form=form, files=[dict(source=inputs.encode(_inspection("rate.npz", values[form], form)))]))
    settings = dict(resolution_x=2., source_resolution_x=1., resolution_t=5., region_start_kb=0.,
                    replicates=3, geometry="periodic" if periodic else "linear", time_window=30.,
                    max_replication_time=60., seed=7, track_time_outputs=True, track_detailed_outputs=True,
                    fork_speed_std=.2, stall_rate_per_kb=.05, stall_duration=2., stall_shared_directions=True)
    recipe = dict(inputs={"initiation_rate": source, "fork_speed": dict(mode="scalar", form="scalar", values=[1.4])},
                  settings=settings, region_end_kb=8., automatic_region_end=8.,
                  export=dict(file_type="NPZ", standard_deviation=True, individual_profiles=True, profile_count=2))
    script = write_script(CodeSnapshot("simulation", recipe, tuple(inputs.files.items())), tmp_path)
    (_case, request), = prepare_simulation_recipe(recipe, input_directory=tmp_path / "inputs")
    expected = run_constant_simulation(request)
    runpy.run_path(str(script), run_name="__main__")
    raw = next(path for path in (tmp_path / "outputs").glob("rsim_rt*.npz") if "standard_deviation" not in path.name)
    with np.load(raw) as data:
        np.testing.assert_array_equal(data["values"].reshape(-1), expected.replication_timing)
        np.testing.assert_array_equal(data["position_kb"], expected.position_kb)
    assert len(tuple((tmp_path / "outputs").glob("*standard_deviation*.npz"))) == 1
    assert len(tuple((tmp_path / "outputs").glob("*_2_replicates.zip"))) == 1


@pytest.mark.parametrize("view", ["Line", "Histogram", "KDE", "Boxplot", "Circular", "Heatmap"])
def test_plotting_script_exports_processed_data_report_and_requested_plot(tmp_path, view):
    from dnascape.workflows.analysis import SmoothingSettings
    from dnascape.workflows.analysis_recipe import prepare_analysis_recipe
    inputs = CodeInputs()
    form = "xt" if view == "Heatmap" else "x"
    values = np.arange(24.).reshape(8, 3) + 1 if form == "xt" else np.arange(12.) + 1
    inspection = _inspection("profile.npz", values, form)
    arguments = dict(plot_type="Heatmap" if form == "xt" else "Histogram" if view == "KDE" else "Line" if view == "Circular" else view,
        title="Code preview", x_label="Position", y_label="Value", color="#2596be", palette="viridis",
        histogram_bins=5, density=view == "KDE", show_legend=True, show_colorbar=True,
        log_y=False, invert_y=False, y_limits=None, series_label="Profile")
    recipe = dict(profiles=[dict(id=[1, "values"], name="Profile", form=form,
        source=inputs.encode(inspection), array_key="values", smoothing=inputs.encode(SmoothingSettings("moving_average", 3)))],
        transformations=dict(settings=inputs.encode(DataFilterSettings(enabled=True, selection="all", action="multiply", value=2.))),
        methods=["pearson", "spearman"], report=dict(file_format="tsv", include_p_values=True),
        plots=[dict(profiles=[[1, "values"]], arguments=arguments,
            curves=[dict(name="Profile", color="#2596be")],
            presentation=dict(view=view, show_fill=True, fill_side="above", reverse_order=True, show_hover=False))])
    snapshot = CodeSnapshot("plotting", recipe, tuple(inputs.files.items()))
    script = write_script(snapshot, tmp_path)
    expected, _report = prepare_analysis_recipe(recipe, input_directory=tmp_path / "inputs")
    runpy.run_path(str(script), run_name="__main__")
    assert (tmp_path / "outputs" / "analysis_report.tsv").is_file()
    html = (tmp_path / "outputs" / "plot_1.html").read_text(encoding="utf-8")
    specification = json.loads(html.split("const specification = ", 1)[1].split(";\n", 1)[0])
    assert specification["tooltip"]["show"] is False
    assert bool(specification.get("dnascapeCircular")) == (view == "Circular")
    np.testing.assert_allclose(expected[0]["values"],
        __import__('dnascape.workflows.analysis', fromlist=['smooth_profile']).smooth_profile(values, SmoothingSettings("moving_average", 3), dependence=form) * 2)


def test_generated_script_runs_from_another_working_directory_without_web_imports(tmp_path):
    inputs = CodeInputs()
    request = DataConversionRequest(_inspection("profile.npz", np.arange(8.)), "x", target_format="npz")
    snapshot = CodeSnapshot("data_manager", dict(sources=[inputs.encode(request)], exports=[{}]), tuple(inputs.files.items()))
    script = write_script(snapshot, tmp_path / "job")
    repository = str(Path(__file__).resolve().parents[2])
    runner = '''import importlib.abc, runpy, sys
sys.path.insert(0, sys.argv[1])
class NoWeb(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'app', 'nicegui'}:
            raise AssertionError('Web dependency: ' + fullname)
sys.meta_path.insert(0, NoWeb())
runpy.run_path(sys.argv[2], run_name='__main__')
'''
    completed = subprocess.run([sys.executable, "-c", runner, repository, str(script)], cwd=tmp_path,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (script.parent / "outputs" / "rdata.npz").is_file()


@pytest.mark.parametrize("page_name", ["data-management", "simulation", "plotting"])
def test_code_button_downloads_current_configuration_without_running_workflow(monkeypatch, tmp_path, page_name):
    import asyncio
    from nicegui.elements.upload import Upload
    from nicegui.testing import user_simulation
    from app import code_export
    from app.export_settings import get_export_settings
    from app.main import index
    from app.pages import data_management, simulation, plotting

    monkeypatch.setattr(get_export_settings(), "dev_mode", True)
    snapshots = []
    original = code_export.build_python

    def capture(snapshot):
        snapshots.append(snapshot)
        return original(snapshot)

    def forbid(*_args, **_kwargs):
        pytest.fail("Exporting code executed the page workflow.")

    monkeypatch.setattr(code_export, "build_python", capture)
    monkeypatch.setattr(data_management, "convert_data", forbid)
    monkeypatch.setattr(simulation, "run_constant_simulation", forbid)
    monkeypatch.setattr(plotting, "_prepare_analysis_profiles", forbid)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-" + page_name).click()

            def control(marker):
                return next(item for item in user.client.elements.values()
                            if marker in item._markers and not item.is_deleted)

            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                for _ in range(5):
                    await asyncio.sleep(0)

            if page_name == "simulation":
                await user.should_see(marker="simulation-region-end", retries=60)
                await set_value("simulation-region-end", 8.)
                await set_value("simulation-replicates", 3)
                await set_value("simulation-fix-seed", True)
                await set_value("simulation-seed", 7)
            else:
                marker = "data-manager-upload" if page_name == "data-management" else "plotting-data-upload"
                await user.should_see(marker=marker, retries=60)
                await control(marker).handle_uploads([
                    Upload.SmallFileUpload("signal.csv", "text/csv", b"position,signal\n0,1\n1,3\n2,2\n3,4\n")
                ])
                await user.should_see("Array size: 4", retries=60)
                if page_name == "data-management":
                    await set_value("data-manager-transform-enabled", True)
                    await set_value("data-manager-transform-formula-1", "source * 2")
                else:
                    await set_value("plotting-line-width", 3.)
                    await set_value("plotting-fill", True)
                    await set_value("plotting-fill-side", "above")
            user.find(marker="action-code").click()
            try:
                download = await user.download.next(timeout=10.)
            except TimeoutError:
                pytest.fail(str(user.notify.messages))
            assert len(snapshots) == 1, user.notify.messages
            assert download.content.startswith(b"# DNAscape")
            assert b"INPUT_DIRECTORY" in download.content
            assert b".zip)" not in download.content
            await user.should_not_see(marker="mapping-code-menu")

    asyncio.run(scenario())
    snapshot = snapshots[0]
    script = write_script(snapshot, tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    assert tuple((tmp_path / "outputs").iterdir())
    if page_name != "plotting":
        assert not tuple((tmp_path / "outputs").glob("*.html"))


@pytest.mark.parametrize("view", ["Default", "Histogram", "KDE", "Boxplot", "Circular", "Separate", "Heatmap"])
def test_plotting_code_preserves_live_curve_and_separate_plot_settings(monkeypatch, tmp_path, view):
    import asyncio
    import copy
    from nicegui.elements.upload import Upload
    from nicegui.testing import user_simulation
    from app import code_export
    from app.export_settings import get_export_settings
    from app.main import index

    monkeypatch.setattr(get_export_settings(), "dev_mode", True)
    snapshots = []
    live = []
    original = code_export.build_python

    def capture(snapshot):
        snapshots.append(snapshot)
        return original(snapshot)

    monkeypatch.setattr(code_export, "build_python", capture)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see(marker="plotting-data-upload", retries=60)

            def control(marker):
                return next(item for item in user.client.elements.values()
                            if marker in item._markers and not item.is_deleted)

            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                for _ in range(6):
                    await asyncio.sleep(0)

            if view == "Heatmap":
                data = io.BytesIO()
                np.savez(data, values=np.arange(24.).reshape(8, 3) + 1, data_form="spacetime")
                files = [Upload.SmallFileUpload("matrix.npz", "application/zip", data.getvalue())]
            else:
                files = [Upload.SmallFileUpload(f"source{i}.csv", "text/csv",
                    ("position,signal\n" + "".join(f"{j},{i + 2 + np.sin(j / 3.)}\n" for j in range(12))).encode()) for i in (1, 2)]
            await control("plotting-data-upload").handle_uploads(files)
            await user.should_see(marker=f"plotting-detected-data-{len(files)}", retries=80)
            for _ in range(100):
                if control("action-plot").enabled:
                    break
                await asyncio.sleep(.05)
            assert control("action-plot").enabled, user.notify.messages
            user.find(marker="action-plot").click()
            try:
                await user.should_see(marker="plotting-result-plot", retries=60)
            except AssertionError:
                pytest.fail(str(user.notify.messages))
            if view == "Separate":
                await set_value("plotting-presentation", "Separate")
                await user.should_see(marker="plotting-result-plot-2", retries=60)
                await set_value("plotting-primary-color-separate-1", "#2596be")
                await set_value("plotting-line-width-separate-1", 3.)
                await set_value("plotting-x-axis-label-separate-2", "Second profile position")
            else:
                if view not in {"Default", "Heatmap"}:
                    await set_value("plotting-view", "Histogram" if view == "KDE" else view)
                if view == "KDE":
                    await set_value("plotting-density", True)
                    await set_value("plotting-kde-smoothness", .7)
                if view == "Default":
                    await set_value("plotting-dual-y-axes", True)
                    await set_value("plotting-y2-axis-label", "Second profile")
                    await set_value("plotting-x2-axis-label", "Second position")
                if view == "Boxplot":
                    await set_value("plotting-box-position-1", 4.)
                    await set_value("plotting-box-label-1", "First source")
                    await set_value("plotting-box-position-2", 8.)
                if view == "Heatmap":
                    await set_value("plotting-z-axis-label", "Signal")
                else:
                    await set_value("plotting-fill", True)
                    await set_value("plotting-fill-side", "above")
                    await set_value("plotting-reverse-order", True)
                await set_value("plotting-show-hover", False)
            live.extend(copy.deepcopy(control("plotting-result-plot" if i == 0 else f"plotting-result-plot-{i + 1}").options)
                        for i in range(2 if view == "Separate" else 1))
            user.find(marker="action-code").click()
            await user.download.next(timeout=10.)

    asyncio.run(scenario())
    assert len(snapshots) == 1
    script = write_script(snapshots[0], tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    for index, reference in enumerate(live, 1):
        html = (tmp_path / "outputs" / f"plot_{index}.html").read_text(encoding="utf-8")
        actual = json.loads(html.split("const specification = ", 1)[1].split(";\n", 1)[0])
        assert len(actual["series"]) == len(reference["series"])
        for rendered, expected in zip(actual["series"], reference["series"], strict=True):
            for key in ("name", "type", "data", "lineStyle", "itemStyle", "areaStyle", "coordinateSystem", "polarIndex", "encode"):
                assert rendered.get(key) == expected.get(key), (view, key)
        for name in ("xAxis", "yAxis"):
            axes = actual[name] if isinstance(actual[name], list) else [actual[name]]
            references = reference[name] if isinstance(reference[name], list) else [reference[name]]
            assert len(axes) == len(references)
            for rendered, expected in zip(axes, references, strict=True):
                for key in ("name", "min", "max", "type", "inverse"):
                    assert rendered.get(key) == expected.get(key), (view, name, key)
