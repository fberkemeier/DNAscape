"""Shared Filter controls constrain Mapping and preserve uploaded inputs."""

import asyncio

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import mapping as page


def test_mapping_filter_controls_fit_export_and_promote_target(monkeypatch):
    results = []
    original = page.run_mapping

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(page, 'run_mapping', record)

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(e for e in user.client.elements.values() if marker in e._markers and not e.is_deleted)

            async def value(marker, new):
                with user.client:
                    control(marker).set_value(new)
                await asyncio.sleep(0)

            await user.open('/')
            user.find(marker='nav-mapping').click()
            await user.should_see(marker='mapping-filter', retries=60)
            assert not control('mapping-filter').value
            assert not control('mapping-filter-enabled').value
            assert control('mapping-filter-relative-to').value == 'replication_timing'
            assert control('mapping-filter-profile-units').value == 'bp'
            await control('source-replication-timing-upload').handle_uploads([
                Upload.SmallFileUpload('timing.tsv', 'text/plain', b'timing\n20\n21\n22\n23\n24\n25\n26\n27\n'),
            ])
            await user.should_see('Array size: 8 x 1', retries=60)
            await value('mapping-fit-initiation-rate-initial', .003)
            await value('mapping-setting-maxiter', 2)
            await value('mapping-filter-threshold', 23.)
            await value('mapping-filter-enabled', True)
            await value('mapping-filter-export', True)
            assert control('mapping-fit-initiation-rate-initial').value == .003
            user.find(marker='action-map').click()
            for _ in range(300):
                await asyncio.sleep(.02)
                if results and control('mapping-filter-enabled').enabled:
                    break
            assert results
            np.testing.assert_array_equal(results[-1].outputs['initiation_rate'][:3], 0.)
            assert control('mapping-filter-enabled').enabled
            assert len([e for e in user.client.elements.values()
                        if 'mapping-filter-raw-mapping-case-1' in e._markers and not e.is_deleted]) == 1
            await value('mapping-target-form', 'time')
            assert control('mapping-target-form').value == 'spacetime'
            assert control('mapping-target-shape').value == 'free'
            await user.should_see(marker='mapping-target-shape')
            await user.should_see('Array size: 8 x 1', retries=60)
            await value('mapping-filter-selection', 'profile')
            assert not control('mapping-filter-relative-to').visible
            await value('mapping-filter-selection', 'threshold')
            await value('mapping-target', 'fork_speed')
            await value('mapping-target-form', 'space')
            assert control('mapping-setting-spatial-knots').value == 12
            assert control('mapping-setting-tolerance').value == 1e-5
            assert not any('mapping-setting-fit-step' in e._markers and not e.is_deleted
                           for e in user.client.elements.values())

    asyncio.run(scenario())
