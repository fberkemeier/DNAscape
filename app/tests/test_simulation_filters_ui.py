"""Region switches and independent filters drive the real Simulation workflow."""

import asyncio

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import simulation as page


def test_simulation_region_switches_both_filters_guides_and_exports(monkeypatch):
    requests, results = [], []
    original = page.run_constant_simulation

    def record(request, *args, **kwargs):
        result = original(request, *args, **kwargs)
        requests.append(request)
        results.append(result)
        return result

    monkeypatch.setattr(page, 'run_constant_simulation', record)

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(e for e in user.client.elements.values() if marker in e._markers and not e.is_deleted)

            async def value(marker, new):
                with user.client:
                    control(marker).set_value(new)
                await asyncio.sleep(0)

            def guide(key):
                suffix = f'-{key.replace("_", "-")}-filter'
                return next(e for e in user.client.elements.values() if not e.is_deleted and
                            any(marker.endswith(suffix) for marker in e._markers))

            async def simulate():
                count = len(results)
                user.find(marker='action-simulate').click()
                for _ in range(350):
                    await asyncio.sleep(.02)
                    if len(results) > count and control('simulation-filter-initiation-rate-enabled').enabled:
                        break
                assert len(results) == count + 1

            await user.open('/')
            user.find(marker='nav-simulation').click()
            await user.should_see(marker='simulation-whole-region', retries=60)
            assert not control('simulation-whole-region').enabled
            assert control('simulation-show-plot').value
            assert not any('simulation-geometry' in e._markers and not e.is_deleted for e in user.client.elements.values())
            await value('source-initiation-rate-mode', 'profile')
            await control('source-initiation-rate-upload').handle_uploads([
                Upload.SmallFileUpload('rates.csv', 'text/csv', b'rate\n0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n'),
            ])
            await value('source-fork-speed-mode', 'profile')
            await control('source-fork-speed-upload').handle_uploads([
                Upload.SmallFileUpload('speed.csv', 'text/csv', b'speed\n1\n2\n3\n4\n5\n6\n'),
            ])
            await user.should_see(marker='source-initiation-rate-detected-data', retries=60)
            await user.should_see(marker='source-fork-speed-detected-data', retries=60)
            for _ in range(100):
                if control('simulation-whole-region').enabled:
                    break
                await asyncio.sleep(.02)
            assert control('simulation-whole-region').enabled
            await value('simulation-whole-region', True)
            assert control('simulation-region-start').value == 0.
            assert control('simulation-region-end').value == 6.
            assert not control('simulation-region-start').enabled
            await value('simulation-periodic', True)
            await value('simulation-show-plot', False)
            await value('simulation-replicates', 2)
            await value('simulation-fix-seed', True)
            await value('simulation-filter-relative-to', 'both')
            for key in ('initiation-rate', 'fork-speed'):
                await value(f'simulation-filter-{key}-enabled', True)
                await value(f'simulation-filter-{key}-export', True)
            await value('simulation-filter-initiation-rate-threshold', .25)
            await value('simulation-filter-fork-speed-comparison', 'above')
            await value('simulation-filter-fork-speed-threshold', 1.5)
            await value('simulation-filter-fork-speed-action', 'multiply')
            await value('simulation-filter-fork-speed-factor', .5)
            await simulate()
            assert results[-1].geometry == 'periodic'
            np.testing.assert_allclose(requests[-1].initiation_rate, [0., 0., .3, .4, .5, .6])
            np.testing.assert_allclose(requests[-1].fork_speed, [1., 1., 1.5, 2., 2.5, 3.])
            assert set(results[-1].filter_state.selections) == {'initiation_rate', 'fork_speed'}
            await user.should_not_see(marker='simulation-result-plot')
            await user.should_see('Initiation-rate mask positions', retries=100)
            await user.should_see('Fork-speed mask positions', retries=100)
            for key in ('initiation-rate', 'fork-speed'):
                assert any(not e.is_deleted and any(m.startswith(f'simulation-{key}-filter-') and m.endswith('-raw') for m in e._markers)
                           for e in user.client.elements.values())
            await value('simulation-show-plot', True)
            await user.should_see(marker='simulation-result-plot', retries=60)
            with user.client:
                guide('initiation_rate').set_value(True)
                guide('fork_speed').set_value(True)
            await asyncio.sleep(.02)
            names = [series['name'] for series in control('simulation-result-plot').options['series']]
            assert 'Initiation-rate mask' in names and 'Fork-speed mask' in names
            with user.client:
                guide('initiation_rate').set_value(False)
            await asyncio.sleep(.02)
            names = [series['name'] for series in control('simulation-result-plot').options['series']]
            assert 'Initiation-rate mask' not in names and 'Fork-speed mask' in names
            await value('simulation-filter-relative-to', 'fork_speed')
            await simulate()
            assert set(results[-1].filter_state.selections) == {'fork_speed'}
            np.testing.assert_allclose(requests[-1].initiation_rate, [.1, .2, .3, .4, .5, .6])

    asyncio.run(scenario())
