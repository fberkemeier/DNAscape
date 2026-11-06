"""Every developer download runs on its own and keeps local inputs editable."""
from __future__ import annotations

import asyncio
import io
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from app.code_export import CodeInputs, CodeSnapshot, build_python, filter_code_settings
from app.mapping_code_export import MappingCodeSnapshot, build_mapping_python
from dnascape.workflows.code_recipes import encode_input_files, prepare_input_files, restore_settings
from dnascape.workflows.conversion import DataConversionRequest
from dnascape.workflows.filtering import DataFilterSettings
from dnascape.workflows.uploads import inspect_upload


def profile_bytes(value=.1):
    stream = io.BytesIO()
    np.savez(stream, values=np.full(8, value), data_form="space")
    return stream.getvalue()


def download(workflow):
    inputs = CodeInputs()
    inspection = inspect_upload("source.npz", profile_bytes(), expected_form="x")
    source = inputs.encode(inspection)
    if workflow == "data_manager":
        recipe = dict(sources=[inputs.encode(DataConversionRequest(
            inspection, "x", target_format="npz", output_name="result"))])
    elif workflow == "simulation":
        recipe = dict(
            inputs=dict(initiation_rate=dict(mode="profile", form="x", files=[dict(source=source)]),
                        fork_speed=dict(mode="scalar", form="scalar", values=[1.4])),
            settings=dict(resolution_x=1., source_resolution_x=1., resolution_t=5.,
                          region_start_kb=0., replicates=2, geometry="linear",
                          time_window=30., max_replication_time=60., seed=7),
            region_end_kb=8., automatic_region_end=8.,
            export=dict(file_type="NPZ", filename="result"),
        )
    elif workflow == "plotting":
        recipe = dict(profiles=[dict(id=[1, "values"], name="Signal", form="x", source=source,
                                    array_key="values")],
                      data_exports={"1:values": dict(name="result", type="NPZ")})
    else:
        recipe = dict(method="firing_and_speed_to_timing_deterministic", target_shape=None,
                      inputs=dict(initiation_rate=dict(form="x", mode="upload", files=[dict(path="source.npz")]),
                                  fork_speed=dict(form="scalar", mode="constant", values=1.4)),
                      region=dict(start_kb=0., end_kb=8.), settings=dict(resolution=1., perQ=False),
                      setting_overrides=dict(neighbourhood_range=dict(enabled=True, value=10.),
                                             n_bins=dict(enabled=False, value=8)))
        return build_mapping_python(MappingCodeSnapshot(
            recipe, dict(output="primary", filename="result", file_format="NPZ"),
            tuple(inputs.files.items()),
        ))
    return build_python(CodeSnapshot(workflow, recipe, tuple(inputs.files.items())))


@pytest.mark.parametrize("workflow", ["data_manager", "mapping", "simulation", "plotting"])
def test_single_download_restores_originals_and_respects_local_replacements(tmp_path, workflow):
    artifact = download(workflow)
    script = tmp_path / artifact.filename
    script.write_bytes(artifact.data)
    assert not (tmp_path / "inputs").exists()
    namespace = runpy.run_path(str(script), run_name="__main__")
    original = tmp_path / "inputs" / "source.npz"
    assert original.read_bytes() == profile_bytes()
    assert (tmp_path / "outputs" / "result.npz").is_file()
    assert b"from app" not in artifact.data and b"nicegui" not in artifact.data

    local = tmp_path / "replacement"
    local.mkdir()
    replacement = profile_bytes(.2)
    (local / "source.npz").write_bytes(replacement)
    namespace["main"](input_directory=local, output_directory=tmp_path / "rerun")
    assert (local / "source.npz").read_bytes() == replacement
    assert (tmp_path / "rerun" / "result.npz").is_file()
    if workflow in {"data_manager", "plotting"}:
        with np.load(tmp_path / "rerun" / "result.npz") as data:
            np.testing.assert_array_equal(data["values"].reshape(-1), np.full(8, .2))


def test_embedded_inputs_follow_edited_recipe_paths_and_never_create_placeholders(tmp_path):
    files = encode_input_files((("source.tsv", b"1\n2\n"), ("missing.tsv", b"")))
    prepare_input_files(files, tmp_path, recipe={"source": {"__upload__": "edited.tsv"}})
    assert not tuple(tmp_path.iterdir())
    prepare_input_files(files, tmp_path, recipe={"source": {"__upload__": "missing.tsv"}})
    assert not tuple(tmp_path.iterdir())


def test_embedded_inputs_preserve_duplicate_basenames_and_stay_inside_input_folder(tmp_path):
    files = encode_input_files((("signal.tsv", b"1\n"), ("source_2/signal.tsv", b"2\n")))
    recipe = [{"path": path} for path in files]
    prepare_input_files(files, tmp_path, recipe=recipe)
    assert (tmp_path / "signal.tsv").read_bytes() == b"1\n"
    assert (tmp_path / "source_2" / "signal.tsv").read_bytes() == b"2\n"
    unsafe = encode_input_files((("../outside.tsv", b"3\n"),))
    with pytest.raises(ValueError, match="inside INPUT_DIRECTORY"):
        prepare_input_files(unsafe, tmp_path, recipe={"path": "../outside.tsv"})


def test_embedded_inputs_can_be_disabled_and_script_runs_in_a_notebook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = download("data_manager")
    namespace = {"__name__": "notebook"}
    exec(compile(artifact.data, artifact.filename, "exec"), namespace)
    assert namespace["HERE"] == tmp_path
    namespace["USE_EMBEDDED_INPUTS"] = False
    with pytest.raises(FileNotFoundError, match="source file"):
        namespace["main"]()
    assert not (tmp_path / "inputs").exists()
    namespace["USE_EMBEDDED_INPUTS"] = True
    namespace["main"]()
    assert (tmp_path / "outputs" / "result.npz").is_file()


def test_disabled_transforms_do_not_require_stale_uploaded_masks():
    controls = SimpleNamespace(snapshot=lambda: DataFilterSettings(
        enabled=False, selection="profile", upload=inspect_upload("old-mask.tsv", b"1\n2\n")))
    inputs = CodeInputs()
    encoded = inputs.encode(filter_code_settings(controls))
    assert not inputs.files
    assert restore_settings(encoded) == DataFilterSettings()


def test_prepared_mapping_bundle_prefers_an_edited_local_file(tmp_path):
    from dnascape.workflows.mapping_recipe import _load_prepared_input

    embedded = {"x_array": np.array([1., 2.])}
    inputs = CodeInputs()
    source = dict(prepared_path="prepared.npz", prepared_values=inputs.encode(embedded))
    np.testing.assert_array_equal(_load_prepared_input(source, tmp_path)["x_array"], [1., 2.])
    np.savez(tmp_path / "prepared.npz", x_array=np.array([3., 4.]))
    np.testing.assert_array_equal(_load_prepared_input(source, tmp_path)["x_array"], [3., 4.])


def test_mapping_settings_have_one_effective_editable_value(tmp_path):
    from dnascape.workflows.mapping_recipe import prepare_mapping_recipe

    artifact = download("mapping")
    script = tmp_path / artifact.filename
    script.write_bytes(artifact.data)
    namespace = runpy.run_path(str(script))
    recipe = namespace["MAPPING"]
    assert recipe["setting_overrides"] == {"n_bins": dict(enabled=False, value=8)}
    recipe["settings"]["neighbourhood_range"] = 23.
    prepare_input_files(namespace["EMBEDDED_INPUTS"], tmp_path, recipe=recipe)
    (_, request), = prepare_mapping_recipe(recipe, base_path=tmp_path)
    assert request.settings["neighbourhood_range"] == 23.
    assert "n_bins" not in request.settings


def test_code_input_limit_counts_original_uploads(monkeypatch):
    from dnascape.workflows import code_recipes

    monkeypatch.setattr(code_recipes, "MAX_CODE_INPUT_BYTES", 2)
    with pytest.raises(ValueError, match="download limit"):
        encode_input_files((("large.tsv", b"123"),))


@pytest.mark.parametrize("page_name", ["data-management", "mapping", "simulation", "plotting"])
def test_live_code_download_is_complete_without_running_the_page(monkeypatch, tmp_path, page_name):
    from nicegui.elements.upload import Upload
    from nicegui.testing import user_simulation

    from app.export_settings import get_export_settings
    from app.main import index
    from app.pages import data_management, mapping, plotting, simulation

    def forbid(*_args, **_kwargs):
        pytest.fail("Downloading Code started a page calculation.")

    monkeypatch.setattr(data_management, "convert_data", forbid)
    monkeypatch.setattr(mapping, "run_mapping", forbid)
    monkeypatch.setattr(simulation, "run_constant_simulation", forbid)
    monkeypatch.setattr(plotting, "_prepare_analysis_profiles", forbid)
    downloads = []

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            with user.client:
                get_export_settings().dev_mode = True
            user.find(marker="nav-" + page_name).click()
            await user.should_see(marker="action-code", retries=60)

            def control(marker):
                return next(item for item in user.client.elements.values()
                            if marker in item._markers and not item.is_deleted)

            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                for _ in range(5):
                    await asyncio.sleep(0)

            marker = {
                "data-management": "data-manager-upload", "mapping": "source-replication-timing-upload",
                "simulation": "source-initiation-rate-upload", "plotting": "plotting-data-upload",
            }[page_name]
            if page_name == "simulation":
                await set_value("simulation-region-end", 8.)
                await set_value("simulation-replicates", 2)
                await set_value("simulation-fix-seed", True)
                await set_value("simulation-seed", 7)
                await set_value("source-initiation-rate-mode", "profile")
            elif page_name == "mapping":
                await set_value("mapping-region-end", 8.)
            await control(marker).handle_uploads([
                Upload.SmallFileUpload("source.npz", "application/octet-stream", profile_bytes())])
            await user.should_see("Array size: 8", retries=60)
            user.find(marker="action-code").click()
            downloads.append(await user.download.next(timeout=15.))

    asyncio.run(scenario())
    artifact = downloads[0]
    script = tmp_path / "download.py"
    script.write_bytes(artifact.content)
    runpy.run_path(str(script), run_name="__main__")
    assert (tmp_path / "inputs" / "source.npz").read_bytes() == profile_bytes()
    assert tuple((tmp_path / "outputs").iterdir())
