"""Mapping distinguishes event counts per length from event rates per length."""

import asyncio

from nicegui.testing import user_simulation

from app.main import index
from app.pages.mapping import _mapping_dependence_label, _OUTPUT_LABELS
from dnascape.workflows.mapping_catalog import mapping_target


def test_event_density_quantity_labels_and_units():
    for name in ("initiation_events", "coalescence_events"):
        target = mapping_target(name)
        assert target.variant("space").label == "Density (events/kb)"
        assert target.variant("spacetime").label == "Rate density (events/kb/min)"
        assert "per kb" in target.variant("space").description
        assert _mapping_dependence_label(name, "spacetime") == "Rate density (events/kb/min)"
        assert "events/kb)" in _OUTPUT_LABELS[name]
        assert "events/kb/min)" in _OUTPUT_LABELS[name+"_xt"]


def test_event_density_quantity_menu_updates_without_changing_other_targets():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            def field(marker):
                return next(e for e in user.find(marker=marker).elements if not e.is_deleted)
            for name in ("initiation_events", "coalescence_events", "initiation_rate"):
                with user.client:
                    field("mapping-target").set_value(name)
                for _ in range(5):
                    await asyncio.sleep(0)
                quantity = field("mapping-target-form")
                if name == "initiation_rate":
                    assert quantity._props["label"] == "Dependence"
                    assert quantity.options["space"] == "Space"
                else:
                    assert quantity._props["label"] == "Quantity"
                    assert quantity.options == {
                        "space": "Density (events/kb)",
                        "spacetime": "Rate density (events/kb/min)",
                    }
                    with user.client:
                        quantity.set_value("spacetime")
                    for _ in range(5):
                        await asyncio.sleep(0)
                    assert quantity.value == "spacetime"
                    assert any(getattr(e, "text", None) == "Rate density (events/kb/min)"
                               for e in user.client.elements.values() if not e.is_deleted)
    asyncio.run(scenario())
