"""Every route exposes one control per fitting phase without breaking aliases."""

from itertools import product

import numpy as np
import pytest

from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import get_mapping_catalog, mapping_route


def test_all_route_form_and_shape_combinations_avoid_redundant_fit_controls():
    for route in get_mapping_catalog().routes:
        for shape in route.target_shapes or (None,):
            for values in product(*(item.forms for item in route.inputs)):
                forms = dict(zip((item.key for item in route.inputs), values))
                keys = {item.key for item in route.active_settings(shape, forms)}
                assert not route.single_phase_fit_aliases.keys() & keys, (route.method, shape, forms)
                for first, second in (("maxiter", "space_maxiter"), ("fit_step", "space_fit_step"),
                                      ("maxiter", "temporal_maxiter"), ("fit_step", "temporal_fit_step")):
                    assert not {first, second} <= keys, (route.method, shape, forms)


@pytest.mark.parametrize("method", [
    "timing_and_speed_to_initiation_rate", "timing_to_firing_constant_fork_speed",
    "timing_and_variable_speed_to_spatial_initiation_rate", "timing_to_spatial_initiation_rate_variable_fork_speed",
])
def test_existing_single_phase_aliases_keep_their_original_effective_precedence(method):
    variable = "variable" in method
    speed = np.linspace(1.2, 1.6, 32) if variable else 1.4
    common = dict(
        route=mapping_route(method), inputs={"replication_timing": np.linspace(40., 60., 32), "fork_speed": speed},
        input_forms={"replication_timing": "x", "fork_speed": "x" if variable else "scalar"},
        resolution_x=2.,
    )
    options = {"sim_number": 8, "refine_seed": 19} if variable else {}
    canonical = run_mapping(MappingRequest(**common, settings={**options, "maxiter": 2, "fit_step": .3}))
    legacy = run_mapping(MappingRequest(**common, settings={
        **options, "maxiter": 7, "fit_step": 1., "space_maxiter": 2, "space_fit_step": .3,
    }))
    np.testing.assert_array_equal(legacy.outputs["initiation_rate"], canonical.outputs["initiation_rate"])
    assert legacy.assumptions["maxiter"] == 2
    assert legacy.assumptions["fit_step"] == .3
    assert not {"space_maxiter", "space_fit_step"} & legacy.assumptions.keys()
