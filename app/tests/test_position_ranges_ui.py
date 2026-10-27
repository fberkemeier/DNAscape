"""Shared interval controls and portable code retain typed genomic masks."""

import asyncio
from dataclasses import replace
import runpy

import numpy as np
import pandas as pd
import pytest
from nicegui.testing import user_simulation

from app.data_filter_controls import DataFilterControls
from app.main import index
from dnascape.workflows.filtering import FilterProfile, resolve_filter


@pytest.mark.parametrize("page,prefix", [
    ("data-management", "data-manager-filter"),
    ("mapping", "mapping-filter"),
    ("simulation", "simulation-filter-initiation-rate"),
    ("simulation", "simulation-filter-fork-speed"),
    ("plotting", "analysis-transform"),
])
def test_ranges_are_shared_expandable_and_independent_of_uploads(monkeypatch, page, prefix):
    created = []
    original = DataFilterControls.__init__

    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(DataFilterControls, "__init__", capture)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker=f"nav-{page}").click()
            if page == "simulation":
                await user.should_see(marker="simulation-filter-relative-to", retries=60)
                with user.client:
                    next(iter(user.find(marker="simulation-filter-relative-to").elements)).set_value("both")
            await user.should_see(marker=prefix, retries=60)
            section = next(item for item in created if item.marker_prefix == prefix)
            controls = section.controls
            choices = list(controls["selection"].options)
            assert choices.index("ranges") + 1 == choices.index("profile")
            assert controls["selection"].options["profile"] == "Upload profile"
            with user.client:
                controls["selection"].set_value("ranges")
            assert section._ranges_group.visible and not section._profile_group.visible
            assert controls["ranges_units"].value == "bp"
            assert controls["ranges"]._props["autogrow"]
            assert controls["ranges"]._props["wrap"] == "soft"
            if "relative_to" in controls:
                assert not controls["relative_to"].visible
            with user.client:
                controls["ranges"].set_value("(1000,2000), (4000,5000)")
            assert section.selection_ready
            settings = resolve_filter(replace(section.snapshot(), enabled=True))
            np.testing.assert_array_equal(settings.profile.values, [[1000, 2000], [4000, 5000]])
            assert settings.profile.units == "bp"
            with user.client:
                controls["ranges_units"].set_value("kb")
            assert resolve_filter(replace(section.snapshot(), enabled=True)).profile.units == "kb"
            # Incomplete text is inert until Apply transform is enabled.
            with user.client:
                controls["ranges"].set_value("(1000,")
            assert not section.selection_ready
            assert not resolve_filter(section.snapshot()).enabled
            with pytest.raises(ValueError, match="position ranges"):
                resolve_filter(replace(section.snapshot(), enabled=True))

    asyncio.run(scenario())


def test_generated_code_uses_typed_ranges_instead_of_a_previous_mask_upload(monkeypatch, tmp_path):
    from app.code_export import CodeInputs, CodeSnapshot, build_python, filter_code_settings
    from dnascape.workflows.conversion import DataConversionRequest
    from dnascape.workflows.uploads import inspect_upload

    created = []
    original = DataFilterControls.__init__

    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(DataFilterControls, "__init__", capture)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="data-manager-filter", retries=60)
            section = created[0]
            section.upload = inspect_upload("old-mask.tsv", b"position_bp\n4000\n", expected_form="regions")
            section.profile = FilterProfile(np.array([4000.]), "positions", "bp")
            with user.client:
                section.controls["selection"].set_value("ranges")
                section.controls["ranges"].set_value("(1000,3000)")
                section.controls["enabled"].set_value(True)
            return filter_code_settings(section)

    settings = asyncio.run(scenario())
    assert settings.selection == "ranges" and settings.upload is None and settings.profile is None
    source = inspect_upload("source.tsv", b"position_kb\tsignal\n0\t10\n1\t11\n2\t12\n3\t13\n4\t14\n5\t15\n", expected_form="x")
    inputs = CodeInputs()
    request = DataConversionRequest(inspection=source, input_form="x", value_column="signal", data_filter=settings)
    recipe = {"sources": [inputs.encode(request)], "exports": [{"output_name": "masked", "target_format": "tsv",
        "filter": {"output_name": "mask", "target_format": "tsv", "layout": "positions"}}]}
    assert list(inputs.files) == ["source.tsv"]
    input_folder = tmp_path / "inputs"
    input_folder.mkdir()
    for name, content in inputs.files.items():
        (input_folder / name).write_bytes(content)
    artifact = build_python(CodeSnapshot("data_manager", recipe, tuple(inputs.files.items())))
    script = tmp_path / artifact.filename
    script.write_bytes(artifact.data)
    runpy.run_path(str(script), run_name="__main__")
    np.testing.assert_array_equal(pd.read_csv(tmp_path / "outputs/masked.tsv", sep="\t").iloc[:, -1], [10, 0, 0, 13, 14, 15])
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "outputs/mask.tsv"), [1000, 2000])
