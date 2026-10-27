"""Live Mapping coverage for filename-aligned constant batches."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation
from nicegui.testing.user import UserInteraction

from app.main import index
from app.mapping_service import MappingRequest, MappingResult

_ROUTE = "initiation_rate--space--replication_timing-and-fork_speed"


def _mapped_result(request: MappingRequest) -> MappingResult:
    values = np.asarray(request.inputs["replication_timing"], dtype=float)
    return MappingResult(
        target=request.route.target,
        target_key=request.route.target_key,
        target_label="Initiation rate",
        method=request.route.method,
        output_form="x",
        output_dependence="space",
        primary_output="initiation_rate",
        outputs={"initiation_rate": values},
        output_forms={"initiation_rate": "x"},
        x_axis=np.arange(values.size, dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
    )


async def _wait_for(predicate: Any, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition did not become true")


async def _open_route(user: Any) -> None:
    await user.open("/")
    user.find(marker="nav-mapping").click()
    await user.should_see(marker="mapping-route", retries=60)
    route = cast(
        Select,
        next(iter(user.find(marker="mapping-route").elements)),
    )
    assert user.client is not None
    with user.client:
        route.set_value(_ROUTE)
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def _use_multiple_constants(user: Any) -> Select:
    input_type = cast(
        Select,
        next(iter(user.find(marker="mapping-input-fork-speed-type").elements)),
    )
    assert user.client is not None
    with user.client:
        input_type.set_value("constants")
    await asyncio.sleep(0)
    await user.should_see(
        marker="mapping-input-fork-speed-combination-mode",
        retries=20,
    )
    return cast(
        Select,
        next(
            iter(
                user.find(
                    marker="mapping-input-fork-speed-combination-mode"
                ).elements
            )
        ),
    )


def _constant(user: Any, index: int) -> Number:
    return cast(
        Number,
        next(
            iter(
                user.find(
                    marker=f"mapping-input-fork-speed-constant-{index}"
                ).elements
            )
        ),
    )


def test_mapping_constants_pairwise_and_all_combinations_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        report(0.8, "Mapping input combination...")
        return _mapped_result(request)

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await _open_route(user)
            uploader = cast(
                Upload,
                next(
                    iter(
                        user.find(
                            marker="source-replication-timing-upload"
                        ).elements
                    )
                ),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing_a.tsv",
                        "text/tab-separated-values",
                        b"value\n8\n7\n9\n",
                    ),
                    Upload.SmallFileUpload(
                        "timing_b.tsv",
                        "text/tab-separated-values",
                        b"value\n6\n5\n7\n8\n",
                    ),
                ]
            )
            await user.should_see("Datasets: 2", retries=40)
            matching = await _use_multiple_constants(user)

            assert matching.value == "pairwise"
            assert matching.options == {
                "pairwise": "Pairwise",
                "all": "All combinations",
            }
            first = _constant(user, 1)
            second = _constant(user, 2)
            assert first._props["aria-label"] == "Fork speed for timing_a.tsv"
            assert second._props["aria-label"] == "Fork speed for timing_b.tsv"
            assert "dn-mapping-constant-value" in first._classes
            assert "dn-mapping-constant-value" in second._classes
            with user.client:
                first.set_value(1.2)
                second.set_value(1.8)

            # Constant lists remain editable when profiles are already uploaded.
            await user.should_see("Fork speed 1")
            await user.should_see("Fork speed 2")
            user.find(marker="mapping-input-fork-speed-constant-add").click()
            await user.should_see(marker="mapping-input-fork-speed-constant-3", retries=30)
            user.find(marker="mapping-input-fork-speed-constant-remove-3").click()
            await user.should_not_see(marker="mapping-input-fork-speed-constant-3", retries=30)
            assert _constant(user, 1).value == pytest.approx(1.2)
            assert _constant(user, 2).value == pytest.approx(1.8)

            user.find(marker="action-map").click()
            await _wait_for(lambda: len(requests) == 2)
            await user.should_see(marker="mapping-result-plot", retries=60)
            await _wait_for(
                lambda: all(
                    getattr(element, "enabled", False)
                    for element in user.find(marker="action-map").elements
                )
            )
            assert [request.inputs["fork_speed"] for request in requests] == pytest.approx(
                [1.2, 1.8]
            )
            assert [
                float(np.asarray(request.inputs["replication_timing"])[0])
                for request in requests
            ] == [8.0, 6.0]
            chart = next(iter(user.find(marker="mapping-result-plot").elements))
            assert [series["name"] for series in chart.options["series"]] == [
                "Replication timing 1 · Fork speed 1",
                "Replication timing 2 · Fork speed 2",
            ]
            output = next(iter(user.find(marker="mapping-preview-output").elements))
            assert output.options["input:replication_timing"] == "Replication timing (input)"
            assert output.options["input:fork_speed"] == "Fork speed (input)"
            assert "all_plots" in output.options
            await user.should_see(marker="mapping-dataset-name-1", retries=30)
            name = next(iter(user.find(marker="mapping-dataset-name-1").elements))
            default_name = name.value
            with user.client:
                name.set_value("First fit")
            await asyncio.sleep(0)
            renamed = next(iter(user.find(marker="mapping-result-plot").elements))
            assert renamed.options["series"][0]["name"] == "First fit"
            with user.client:
                name.set_value("")
            await asyncio.sleep(0)
            restored = next(iter(user.find(marker="mapping-result-plot").elements))
            assert restored.options["series"][0]["name"] == default_name
            with user.client:
                output.set_value("input:replication_timing")
            await asyncio.sleep(0)
            chart = next(iter(user.find(marker="mapping-result-plot").elements))
            assert [point[1] for point in chart.options["series"][0]["data"]] == [8.0, 7.0, 9.0]
            layout = next(iter(user.find(marker="mapping-preview-layout").elements))
            assert layout.value == "Combined"
            with user.client:
                layout.set_value("Individual")
            await asyncio.sleep(0)

            matching = cast(
                Select,
                next(
                    iter(
                        user.find(
                            marker="mapping-input-fork-speed-combination-mode"
                        ).elements
                    )
                ),
            )
            with user.client:
                matching.set_value("all")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await _wait_for(
                lambda: all(
                    getattr(element, "enabled", False)
                    for element in user.find(marker="action-map").elements
                )
            )
            user.find(marker="action-map").click()
            await _wait_for(lambda: len(requests) == 6)
            all_cases = [
                (
                    float(np.asarray(request.inputs["replication_timing"])[0]),
                    float(request.inputs["fork_speed"]),
                )
                for request in requests[2:]
            ]
            assert all_cases == [
                (8.0, 1.2),
                (8.0, 1.8),
                (6.0, 1.2),
                (6.0, 1.8),
            ]
            await _wait_for(lambda: layout.value == "Combined")

    asyncio.run(scenario())


def test_removing_middle_profile_keeps_constants_bound_to_filename() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await _open_route(user)
            uploader = cast(
                Upload,
                next(
                    iter(
                        user.find(
                            marker="source-replication-timing-upload"
                        ).elements
                    )
                ),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        f"timing_{letter}.tsv",
                        "text/tab-separated-values",
                        f"value\n{index}\n{index + 1}\n".encode(),
                    )
                    for index, letter in enumerate(("a", "b", "c"), start=1)
                ]
            )
            await user.should_see("Datasets: 3", retries=40)
            await _use_multiple_constants(user)
            controls = [_constant(user, index) for index in range(1, 4)]
            assert user.client is not None
            with user.client:
                for control, value in zip(controls, (1.1, 1.2, 1.3), strict=True):
                    control.set_value(value)

            remove_buttons = sorted(
                user.find(
                    marker="source-replication-timing-upload-remove"
                ).elements,
                key=lambda element: element.id,
            )
            assert len(remove_buttons) == 3
            UserInteraction(user, {remove_buttons[1]}, None).click()
            await user.should_not_see("timing_b.tsv", retries=30)
            await _wait_for(
                lambda: len(
                    user.find(
                        marker="mapping-input-fork-speed-constant-list"
                    ).elements
                )
                == 1
            )
            first = _constant(user, 1)
            second = _constant(user, 2)
            assert first._props["aria-label"] == "Fork speed for timing_a.tsv"
            assert second._props["aria-label"] == "Fork speed for timing_c.tsv"
            assert first.value == pytest.approx(1.1)
            assert second.value == pytest.approx(1.3)

    asyncio.run(scenario())
