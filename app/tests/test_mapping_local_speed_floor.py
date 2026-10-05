"""Mapping exposes and applies local-speed completion conditioning."""

import numpy as np
import pytest

from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping
from app.pages.mapping import _setting_value


@pytest.mark.parametrize("policy", ("mask", "zero", "raise"))
def test_mapping_local_speed_default_controls_and_output_masks_agree(policy):
    route = mapping_route("replicated_fraction_and_fork_density_to_speed_local_inverse")
    forms = {"replicated_fraction_xt": "xt", "fork_density_xt": "xt"}
    active = {item.key: item for item in route.active_settings("best", forms)}
    assert _setting_value(active["unreplicated_floor"]) == .001
    assert _setting_value(active["density_floor"]) == 1e-10
    fraction = np.tile([0., .5, .998, .9995, 1.], (3, 1))
    request = MappingRequest(
        route=route, input_forms=forms,
        inputs={"replicated_fraction_xt": fraction, "fork_density_xt": np.ones_like(fraction)},
        settings={"invalid_policy": policy},
    )
    if policy == "raise":
        with pytest.raises(ValueError, match="unidentifiable"):
            run_mapping(request)
        return
    result = run_mapping(request)
    output = result.outputs[result.primary_output]
    assert np.isfinite(output[:, :3]).all()
    assert np.isnan(output[:, 3:]).all() if policy == "mask" else np.all(output[:, 3:] == 0)
    assert result.diagnostics["unreplicated_floor"] == .001
    assert result.diagnostics["unidentifiable_count"] == 6
    np.testing.assert_array_equal(result.diagnostics["valid_mask"], [[True, True, True, False, False]] * 3)
    assert not result.diagnostics["valid_mask"].flags.writeable


def test_mapping_local_speed_explicit_zero_is_forwarded_without_changing_other_inverses():
    route = mapping_route("replicated_fraction_and_fork_density_to_speed_local_inverse")
    fraction = np.tile([0., .5, .9995, 1.], (3, 1))
    result = run_mapping(MappingRequest(
        route=route,
        input_forms={"replicated_fraction_xt": "xt", "fork_density_xt": "xt"},
        inputs={"replicated_fraction_xt": fraction, "fork_density_xt": np.ones_like(fraction)},
        settings={"unreplicated_floor": 0.},
    ))
    assert np.isfinite(result.outputs[result.primary_output][:, 2]).all()
    assert np.isnan(result.outputs[result.primary_output][:, 3]).all()
    assert result.diagnostics["unreplicated_floor"] == 0.
    firing = mapping_route("initiation_events_and_fraction_to_firing_rate_local_inverse")
    assert firing.setting("unreplicated_floor").default == 1e-8
