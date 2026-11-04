"""Time samples in one row stay together through upload and Mapping."""

import asyncio
import io
from dataclasses import replace

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingRequest, MappingResult, run_mapping, validate_mapping_request
from app.mapping_catalog import mapping_route_families
from dnascape.workflows.uploads import inspect_upload, materialize_upload


@pytest.mark.parametrize("column", [None, "t0", "__all__"])
@pytest.mark.parametrize("source_format", ["tsv", "npz"])
def test_temporal_row_keeps_all_samples_even_with_retained_column_choice(column, source_format):
    values = np.linspace(.001, .01, 301).reshape(1, -1)
    headers = [f"t{i}" for i in range(values.size)]
    if source_format == "tsv":
        content = ("\t".join(headers) + "\n" + "\t".join(map(str, values[0])) + "\n").encode()
    else:
        buffer = io.BytesIO()
        np.savez(buffer, values=values, columns=headers, semantic_form="time")
        content = buffer.getvalue()
    inspection = inspect_upload(f"profile.{source_format}", content, expected_form="auto")
    assert inspection.inferred_form == "t"
    result = materialize_upload(replace(inspection, expected_form="t"), column=column)
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize("target", ["fork_speed", "initiation_rate"])
@pytest.mark.parametrize("dependence", ["constant", "space", "time", "spacetime"])
def test_temporal_kinetic_upload_is_one_complete_input_for_every_target_dependence(
    monkeypatch, target, dependence,
):
    from app.pages import mapping as page

    other = "initiation_rate" if target == "fork_speed" else "fork_speed"
    source_marker = other.replace("_", "-")
    values = np.linspace(.001, .005, 301) if other == "initiation_rate" else np.linspace(1., 2., 301)
    n = 8
    requests = []
    form = dict(constant="scalar", space="x", time="t", spacetime="xt")[dependence]

    def fake_run(request, report, cancel):
        prepared = validate_mapping_request(request)
        requests.append(request)
        np.testing.assert_array_equal(prepared.source[other], values.reshape(1, -1))
        output = {"scalar": np.array(1.), "x": np.ones(n),
                  "t": np.ones((1, 2)), "xt": np.ones((n, 2))}[form]
        report(1., "Complete")
        return MappingResult(
            target=request.route.target, target_key=target, target_label=target,
            method=request.route.method, output_form=form, output_dependence=dependence,
            primary_output=request.route.target, outputs={request.route.target: output},
            output_forms={request.route.target: form}, x_axis=np.arange(n, dtype=float),
            time_axis=np.array([0., 1.]), region_labels=(), elapsed_seconds=.01,
            assumptions={},
        )

    monkeypatch.setattr(page, "run_mapping", fake_run)
    original_read = page._read_mapping_input_batch_async

    async def read_with_retained_choice(key, live, **kwargs):
        if key == other:
            for selected in live.surface["selection_values"].values():
                selected["column"] = "__all__" if target == "fork_speed" else "t0"
        return await original_read(key, live, **kwargs)

    monkeypatch.setattr(page, "_read_mapping_input_batch_async", read_with_retained_choice)

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(element for element in user.client.elements.values()
                            if marker in element._markers and not element.is_deleted)

            async def select(marker, value):
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            async def upload(key, content):
                await user.should_see(marker=f"source-{key}-upload", retries=60)
                await control(f"source-{key}-upload").handle_uploads([
                    Upload.SmallFileUpload(f"{key}.tsv", "text/tab-separated-values", content)
                ])
                await user.should_see(marker=f"mapping-input-{key}-detected-data", retries=60)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await select("mapping-target", target)
            await select("mapping-target-form", dependence)
            await select(f"mapping-input-{source_marker}-type", "profile")
            await upload(source_marker, (
                "\t".join(f"t{i}" for i in range(values.size)) + "\n"
                + "\t".join(map(str, values)) + "\n"
            ).encode())
            assert control(f"mapping-input-{source_marker}-form").value == "t"
            await user.should_not_see(marker=f"mapping-input-{source_marker}-value-column")
            await upload("replication-timing", b"timing\n20\n21\n22\n23\n24\n25\n26\n27\n")
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=100)
            assert len(requests) == 1
            np.testing.assert_array_equal(requests[0].inputs[other], values.reshape(1, -1))

    asyncio.run(scenario())


@pytest.mark.parametrize("target", ["fork_speed", "initiation_rate"])
@pytest.mark.parametrize("dependence", ["constant", "space", "time", "spacetime"])
def test_real_mapping_accepts_complete_temporal_rows(target, dependence):
    samples = 41
    other = "initiation_rate" if target == "fork_speed" else "fork_speed"
    family = next(
        family for family in mapping_route_families(target, dependence)
        if set(family.factor_keys) == {"replication_timing", other}
    )
    route = family.resolve({"replication_timing": "x", other: "t"})
    active = {setting.key for setting in route.active_settings()}
    settings = {key: 1 for key in ("maxiter", "space_maxiter", "temporal_maxiter") if key in active}
    if "resolution_time" in active:
        settings["resolution_time"] = 2.
    values = ((.01 if other == "initiation_rate" else 1.4)
              * (1 + np.arange(samples) / 80)).reshape(1, -1)
    request = MappingRequest(
        route=route, inputs={"replication_timing": np.linspace(20., 25., 8), other: values},
        input_forms={"replication_timing": "x", other: "t"}, settings=settings,
        whole_genome=True, resolution_x=1., resolution_t=2.,
    )
    prepared = validate_mapping_request(request)
    np.testing.assert_array_equal(prepared.source[other], values)
    result = run_mapping(request)
    assert result.output_dependence == dependence
    assert np.all(np.isfinite(result.outputs[result.primary_output]))


def test_temporal_row_missing_values_interpolate_across_time():
    inspection = inspect_upload("profile.tsv", b"t0\tt1\tt2\n1\t\t3\n", expected_form="t")
    np.testing.assert_array_equal(
        materialize_upload(inspection, column="t0", missing_policy="interpolate"), [[1., 2., 3.]],
    )


def test_long_temporal_table_keeps_genuine_value_column_choices():
    inspection = inspect_upload(
        "profiles.tsv", b"time\tspeed_a\tspeed_b\n0\t1\t2\n1\t3\t4\n", expected_form="t",
    )
    np.testing.assert_array_equal(materialize_upload(inspection, column="speed_b"), [[2., 4.]])
