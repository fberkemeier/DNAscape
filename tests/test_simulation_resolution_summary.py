"""Preview spacing must not depend on optional time-course retention."""
import asyncio
from dataclasses import replace

import numpy as np
import pytest
from nicegui.testing import user_simulation

from app.main import index
from app.pages.simulation import _resolution_summary
from dnascape.workflows.simulation import ConstantSimulationRequest, run_constant_simulation


@pytest.mark.parametrize("track_time,track_detailed", [(False, False), (True, False), (False, True)])
def test_configured_resolution_survives_optional_tracking(track_time, track_detailed):
    result = run_constant_simulation(ConstantSimulationRequest(
        initiation_rate=.1, fork_speed=1.4, resolution_x=2., resolution_t=7.5,
        region_start_kb=0., region_end_kb=8., replicates=1, geometry="linear",
        time_window=30., max_replication_time=1200.,
        track_time_outputs=track_time, track_detailed_outputs=track_detailed,
    ))
    assert result.resolution_x == 2.
    assert result.resolution_t == 7.5
    assert _resolution_summary(result) == "2 kb/bin | 7.5 min/bin"
    if not (track_time or track_detailed):
        assert result.time_grid.size == 0
    legacy = replace(result, resolution_x=None, resolution_t=None,
                     time_grid=np.array([]))
    assert _resolution_summary(legacy) == "2 kb/bin"
    singleton = replace(result, position_kb=np.array([0.]), time_grid=np.array([]))
    assert _resolution_summary(singleton) == "2 kb/bin | 7.5 min/bin"


def test_untracked_preview_retains_run_spacing_after_navigation():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="source-initiation-rate-mode", retries=60)

            def set_value(marker, value):
                with user.client:
                    control = next(item for item in user.client.elements.values()
                                   if marker in item._markers and not item.is_deleted)
                    control.set_value(value)

            set_value("source-initiation-rate-mode", "scalar")
            set_value("simulation-region-end", 8.)
            set_value("simulation-resolution-t", 7.5)
            user.find(marker="action-simulate").click()
            await user.should_see("1 kb/bin | 7.5 min/bin", retries=120)
            set_value("simulation-resolution-t", 2.)
            await user.should_see("1 kb/bin | 7.5 min/bin")
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="nav-simulation")
            user.find(marker="nav-simulation").click()
            await user.should_see("1 kb/bin | 7.5 min/bin", retries=60)
    asyncio.run(scenario())
