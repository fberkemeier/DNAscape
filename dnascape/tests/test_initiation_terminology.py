"""Initiation rate is canonical, with legacy names accepted only on input."""

import numpy as np
import pytest

from dnascape import mapt, rmap, rsim, simulate_replication
from dnascape._terminology import canonical_recipe
from dnascape.workflows.mapping_catalog import mapping_route, mapping_target
from dnascape.workflows.mapping_recipe import prepare_mapping_recipe
from dnascape.workflows.result_naming import output_file_token


@pytest.mark.parametrize("key", ["initiation_rate", "ir", "firing_rate", "fr"])
def test_rate_input_aliases_produce_identical_mapping_and_simulation(key):
    expected = rmap(source={"initiation_rate": .01, "fork_speed": 1.4},
                    target="replication_timing", assumptions={"n_bins": 4})
    actual = rmap(source={key: .01, "fork_speed": 1.4},
                  target="replication_timing", assumptions={"n_bins": 4})
    np.testing.assert_array_equal(actual, expected)
    expected_sim = rsim(initiation_rate=.01, n_bins=4, seed=7, verbose=False)
    actual_sim = rsim(source={key: .01}, n_bins=4, seed=7, verbose=False)
    np.testing.assert_array_equal(actual_sim["replication_timing"], expected_sim["replication_timing"])
    assert mapt[key] == "initiation_rate"
    assert output_file_token(key) == "ir"
    assert mapping_target(key).key == "initiation_rate"


def test_old_simulation_keyword_is_read_without_changing_the_canonical_signature():
    import inspect
    assert "initiation_rate" in inspect.signature(rsim).parameters
    assert "firing_rate" not in inspect.signature(rsim).parameters
    expected = rsim(initiation_rate=.01, n_bins=4, seed=7, verbose=False)
    actual = rsim(firing_rate=.01, n_bins=4, seed=7, verbose=False)
    np.testing.assert_array_equal(actual["replication_timing"], expected["replication_timing"])
    expected = simulate_replication(initiation_rate=.01, n_bins=4, rng=np.random.default_rng(7))
    actual = simulate_replication(firing_rate=.01, n_bins=4, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(actual[0], expected[0])
    with pytest.raises(ValueError, match="only once"):
        rsim(initiation_rate=.01, firing_rate=.02, n_bins=4, verbose=False)


def test_old_recipe_keys_change_without_renaming_existing_source_files():
    old = {
        "method": "timing_and_speed_to_firing_rate",
        "settings": {"initial_firing_rate": .01},
        "region": {"firing_rate_per_bin": True},
        "inputs": {"firing_rate": {"files": [{"path": "rdata_fr.tsv", "column": "firing_rate"}]}},
        "filter": {"relative_to": "firing_rate"},
    }
    updated = canonical_recipe(old)
    assert updated["method"] == "timing_and_speed_to_initiation_rate"
    assert updated["settings"] == {"initial_initiation_rate": .01}
    assert updated["region"] == {"initiation_rate_per_bin": True}
    assert updated["filter"]["relative_to"] == "initiation_rate"
    assert updated["inputs"]["initiation_rate"]["files"] == old["inputs"]["firing_rate"]["files"]
    assert mapping_route(old["method"]).method == updated["method"]
    recipe = {"method": "firing_and_speed_to_timing_deterministic",
              "settings": {"resolution": 1.}, "region": {"start_kb": 0., "end_kb": 4.},
              "inputs": {"firing_rate": {"form": "scalar", "mode": "constant", "values": .01},
                         "fork_speed": {"form": "scalar", "mode": "constant", "values": 1.4}}}
    requests = prepare_mapping_recipe(recipe)
    assert set(requests[0][1].inputs) == {"initiation_rate", "fork_speed"}


def test_rate_normalization_preserves_case_sensitive_mapping_settings():
    from dnascape.routing import _canonical_assumption
    assert _canonical_assumption("T0") == "T0"
    assert _canonical_assumption("initial_firing_rate") == "initial_initiation_rate"
