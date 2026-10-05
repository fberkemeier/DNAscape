"""Live Simulation coverage for Mapping-style constant combinations."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation
from nicegui.testing.user import UserInteraction

from app.main import index
from app.mapping_uploads import UploadInspection
from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult


def _result_for(request: ConstantSimulationRequest) -> ConstantSimulationResult:
    bins = int(
        np.ceil((request.region_end_kb - request.region_start_kb) / request.resolution_x)
    )
    position = request.region_start_kb + np.arange(bins, dtype=float) * request.resolution_x
    timing = np.full(bins, float(np.asarray(request.firing_rate).reshape(-1)[0]))
    return ConstantSimulationResult(
        position_kb=position,
        replication_timing=timing,
        elapsed_seconds=0.01,
        replicates=request.replicates,
        geometry=request.geometry,
        time_grid=np.asarray([], dtype=float),
        outputs={"replication_timing": timing},
    )


async def _wait_for(predicate: Any, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition did not become true")


async def _open_simulation(user: Any) -> None:
    await user.open("/")
    user.find(marker="nav-simulation").click()
    await user.should_see(marker="source-firing-rate-mode", retries=60)


def _select(user: Any, marker: str) -> Select:
    return cast(Select, next(iter(user.find(marker=marker).elements)))


def _number(user: Any, factor: str, index: int) -> Number:
    return cast(
        Number,
        next(
            iter(
                user.find(marker=f"source-{factor}-constant-{index}").elements
            )
        ),
    )


async def _use_multiple_constants(user: Any, factor: str) -> None:
    control = _select(user, f"source-{factor}-mode")
    assert user.client is not None
    with user.client:
        control.set_value("constants")
    await user.should_see(marker=f"source-{factor}-constant-1", retries=30)


def _click_add(user: Any, factor: str) -> None:
    user.find(marker=f"source-{factor}-constant-add").click()


def test_simulation_constant_rows_share_the_matching_selection() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await _open_simulation(user)
            await _use_multiple_constants(user, "firing-rate")
            await _use_multiple_constants(user, "fork-speed")

            firing_matching = _select(
                user,
                "source-firing-rate-combination-mode",
            )
            speed_matching = _select(
                user,
                "source-fork-speed-combination-mode",
            )
            assert firing_matching.options == {
                "pairwise": "Pairwise",
                "all": "All combinations",
            }
            assert speed_matching.options == firing_matching.options
            assert firing_matching.value == speed_matching.value == "pairwise"
            assert "dn-mapping-constant-value" in _number(
                user, "firing-rate", 1
            )._classes
            assert _number(user, "firing-rate", 2)._props["aria-label"] == (
                "Firing rate for Value 2"
            )

            assert user.client is not None
            with user.client:
                firing_matching.set_value("all")
            await _wait_for(
                lambda: _select(
                    user,
                    "source-fork-speed-combination-mode",
                ).value
                == "all"
            )

    asyncio.run(scenario())


def test_simulation_pairwise_broadcast_and_all_combinations_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page

    requests: list[ConstantSimulationRequest] = []

    def fake_run(
        request: ConstantSimulationRequest,
        report: Any,
        _cancel: Any,
    ) -> ConstantSimulationResult:
        requests.append(request)
        report(0.8, "Running input combination...")
        return _result_for(request)

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await _open_simulation(user)
            await _use_multiple_constants(user, "firing-rate")

            assert user.client is not None
            with user.client:
                _number(user, "firing-rate", 1).set_value(0.01)
                _number(user, "firing-rate", 2).set_value(0.02)
                # The ordinary Constant mode is the singleton source here.
                next(
                    iter(user.find(marker="source-fork-speed-constant").elements)
                ).set_value(1.4)

            # Pairwise is the default, and a singleton source is broadcast.
            user.find(marker="action-simulate").click()
            await _wait_for(lambda: len(requests) == 2)
            await user.should_see("2 datasets complete", retries=60)
            assert [request.firing_rate for request in requests] == pytest.approx(
                [0.01, 0.02]
            )
            assert [request.fork_speed for request in requests] == pytest.approx(
                [1.4, 1.4]
            )

            await _use_multiple_constants(user, "fork-speed")
            _click_add(user, "fork-speed")
            await user.should_see(marker="source-fork-speed-constant-3", retries=20)
            with user.client:
                _number(user, "fork-speed", 1).set_value(1.1)
                _number(user, "fork-speed", 2).set_value(1.2)
                _number(user, "fork-speed", 3).set_value(1.3)
                _select(
                    user,
                    "source-firing-rate-combination-mode",
                ).set_value("all")
            await _wait_for(
                lambda: _select(
                    user,
                    "source-fork-speed-combination-mode",
                ).value
                == "all"
            )
            await _wait_for(
                lambda: all(
                    getattr(element, "enabled", False)
                    for element in user.find(marker="action-simulate").elements
                )
            )

            user.find(marker="action-simulate").click()
            await _wait_for(lambda: len(requests) == 8)
            assert [
                (float(request.firing_rate), float(request.fork_speed))
                for request in requests[2:]
            ] == pytest.approx(
                [
                    (0.01, 1.1),
                    (0.01, 1.2),
                    (0.01, 1.3),
                    (0.02, 1.1),
                    (0.02, 1.2),
                    (0.02, 1.3),
                ]
            )

    asyncio.run(scenario())


def test_simulation_constants_follow_counterpart_files_and_survive_middle_removal() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await _open_simulation(user)
            await _use_multiple_constants(user, "firing-rate")
            _click_add(user, "firing-rate")
            await user.should_see(marker="source-firing-rate-constant-3", retries=20)
            assert user.client is not None
            with user.client:
                for control, value in zip(
                    (_number(user, "firing-rate", index) for index in range(1, 4)),
                    (0.011, 0.022, 0.033),
                    strict=True,
                ):
                    control.set_value(value)
                _select(user, "source-fork-speed-mode").set_value("profile")
            await user.should_see(marker="source-fork-speed-upload", retries=20)

            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-fork-speed-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        f"speed_{letter}.tsv",
                        "text/tab-separated-values",
                        f"fork_speed\n{value}\n{value}\n".encode(),
                    )
                    for letter, value in zip(
                        ("a", "b", "c"),
                        (1.1, 1.2, 1.3),
                        strict=True,
                    )
                ]
            )
            await user.should_see("Files: 3", retries=60)
            await _wait_for(
                lambda: _number(user, "firing-rate", 3)._props.get("aria-label")
                == "Firing rate for speed_c.tsv"
            )
            controls = [_number(user, "firing-rate", index) for index in range(1, 4)]
            assert [control._props["aria-label"] for control in controls] == [
                "Firing rate for speed_a.tsv",
                "Firing rate for speed_b.tsv",
                "Firing rate for speed_c.tsv",
            ]
            assert [control.value for control in controls] == pytest.approx(
                [0.011, 0.022, 0.033]
            )

            remove_buttons = sorted(
                user.find(marker="source-fork-speed-upload-remove").elements,
                key=lambda element: element.id,
            )
            assert len(remove_buttons) == 3
            UserInteraction(user, {remove_buttons[1]}, None).click()
            await user.should_not_see("speed_b.tsv", retries=30)
            await _wait_for(
                lambda: len(
                    user.find(marker="source-firing-rate-constant-list").elements
                )
                == 1
                and _number(user, "firing-rate", 2)._props.get("aria-label")
                == "Firing rate for speed_c.tsv"
            )
            remaining = [_number(user, "firing-rate", index) for index in (1, 2)]
            assert [control.value for control in remaining] == pytest.approx(
                [0.011, 0.033]
            )

    asyncio.run(scenario())


def test_simulation_case_cap_is_checked_before_cartesian_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.pages.simulation import _plan_simulation_cases, _PreparedSimulationSource

    values = tuple(float(index) for index in range(224))
    first = _PreparedSimulationSource(
        label="Firing rate",
        values=values,
        components=tuple(f"firing-{index}" for index in range(len(values))),
    )
    second = _PreparedSimulationSource(
        label="Fork speed",
        values=values,
        components=tuple(f"speed-{index}" for index in range(len(values))),
    )

    def should_not_materialize(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Cartesian indices were materialized before the cap check")

    monkeypatch.setattr(simulation_page, "plan_combination_indices", should_not_materialize)

    with pytest.raises(ValueError, match=r"50,176 simulation runs.*50,000 combinations"):
        _plan_simulation_cases((first, second), combination_mode="all")


def test_simulation_case_export_stems_remain_unique_after_ascii_folding() -> None:
    from app.pages.simulation import _plan_simulation_cases, _PreparedSimulationSource

    firing = _PreparedSimulationSource(
        label="Firing rate",
        values=(0.01, 0.02),
        components=("数据一", "数据二"),
    )
    speed = _PreparedSimulationSource(
        label="Fork speed",
        values=(1.4,),
        components=("1.4 kb/min",),
    )

    cases = _plan_simulation_cases((firing, speed), combination_mode="pairwise")

    assert len({case.export_stem for case in cases}) == 2
    assert cases[1].export_stem == f"{cases[0].export_stem}-2"


def test_simulation_rejects_mixed_vector_and_spacetime_files_before_running() -> None:
    from app.pages.simulation import _validate_simulation_batch_selection

    controls = SimpleNamespace(
        mode=SimpleNamespace(value="profile"),
        form=SimpleNamespace(value="x"),
        inspections=[
            UploadInspection(
                name="vector.tsv",
                content_type="text/tab-separated-values",
                content=b"value\n1\n2\n",
                source_format="table",
                compressed=False,
                inferred_form="x",
            ),
            UploadInspection(
                name="matrix.tsv",
                content_type="text/tab-separated-values",
                content=b"a\tb\n1\t2\n",
                source_format="table",
                compressed=False,
                inferred_form="xt",
            ),
        ],
        selection={"column": None, "array_key": None},
        selections={},
    )

    with pytest.raises(ValueError, match="one consistent data type"):
        _validate_simulation_batch_selection(controls, label="Firing rate")
