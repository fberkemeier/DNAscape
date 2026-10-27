"""Input previews keep the exact profile values and axes used by each run."""

from types import SimpleNamespace

import numpy as np
import pytest

import dnascape

from app.pages.simulation import _result_plot_options
from dnascape.workflows.exports import select_result_output
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route
from dnascape.workflows.mapping_exports import select_mapping_output
from dnascape.workflows.preview_inputs import add_simulation_input_previews
from dnascape.workflows.plotting import mapping_plot_options
from dnascape.workflows.simulation import ConstantSimulationResult


@pytest.mark.parametrize("profile_speed", (False, True))
def test_mapping_keeps_aligned_profiles_and_broadcasts_constant_inputs(
    profile_speed: bool,
) -> None:
    firing = np.linspace(0.001, 0.003, 30)
    speed = np.linspace(1.2, 1.6, 30) if profile_speed else 1.4
    request = MappingRequest(
        route=mapping_route("firing_to_timing_deterministic"),
        inputs={"initiation_rate": firing, "fork_speed": speed},
        input_forms={"initiation_rate": "x", "fork_speed": "x" if profile_speed else "scalar"},
        settings={"neighbourhood_range": 10.0, "perQ": False},
        region_start_kb=20.0, region_end_kb=24.0, resolution_x=1.0,
    )
    result = run_mapping(request)
    assert result.primary_output == "replication_timing"
    if profile_speed:
        np.testing.assert_array_equal(result.outputs["input:fork_speed"], speed[20:24])
    else:
        np.testing.assert_array_equal(result.outputs["input:fork_speed"], np.full(4, 1.4))
    profile = select_mapping_output(result, "input:initiation_rate")
    np.testing.assert_array_equal(profile.values, firing[20:24])
    np.testing.assert_array_equal(profile.position_kb, [20, 21, 22, 23])
    assert result.output_forms["input:initiation_rate"] == "x"
    assert not result.outputs["input:initiation_rate"].flags.writeable


def test_mapping_spatial_input_keeps_position_axis_when_target_is_temporal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dnascape, "rmap", lambda **_kwargs: {
        "values": np.asarray([1.2, 1.4, 1.6]), "time_grid": np.asarray([0, 5, 10]),
    })
    result = run_mapping(MappingRequest(
        route=mapping_route("timing_and_firing_to_time_fork_speed"),
        inputs={"replication_timing": np.arange(14.0), "initiation_rate": 0.001},
        input_forms={"replication_timing": "x", "initiation_rate": "scalar"},
        region_start_kb=10.0, region_end_kb=14.0, resolution_t=5.0,
    ))
    assert result.output_dependence == "time"
    options = mapping_plot_options(
        result, output_key="input:replication_timing", plot_type="Line", normalization="None",
        title="", x_label="Position (kb)", y_label="Replication timing", color="#b03412",
        palette="viridis", histogram_bins=10, line_width=1.8, show_markers=False,
        show_colorbar=True, log_y=False, invert_y=False,
    )
    np.testing.assert_array_equal(
        options["series"][0]["data"], np.column_stack((np.arange(10, 14), np.arange(10, 14))),
    )
    assert result.output_dependence == "time"


def test_mapping_temporal_input_heatmap_spans_the_selected_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dnascape, "rmap", lambda **_kwargs: {
        "values": np.asarray([4.0, 5.0, 6.0]), "time_grid": np.arange(10.0),
    })
    result = run_mapping(MappingRequest(
        route=mapping_route("firing_to_timing_deterministic"),
        inputs={"initiation_rate": np.asarray([0.1, 0.2, 0.3, 0.4])},
        input_forms={"initiation_rate": "t"}, region_start_kb=30.0, region_end_kb=36.0,
        resolution_x=2.0, resolution_t=3.0,
    ))
    selected = select_mapping_output(result, "input:initiation_rate")
    np.testing.assert_array_equal(selected.position_kb, [30, 32, 34])
    np.testing.assert_array_equal(selected.time_min, [0, 3, 6, 9])
    options = mapping_plot_options(
        result, output_key="input:initiation_rate", plot_type="Heatmap", normalization="None",
        title="", x_label="Position (kb)", y_label="Time (min)", color="#b03412",
        palette="viridis", histogram_bins=10, line_width=1.8, show_markers=False,
        show_colorbar=True, log_y=False, invert_y=False,
    )
    assert options["series"][0]["data"] == [[30, 0, 34, 9]]


@pytest.mark.parametrize(
    ("field", "form", "expected_kind"),
    (
        (np.asarray([0.1, 0.2, 0.3]), "x", "space"),
        (np.asarray([[0.1, 0.2, 0.3, 0.4]]), "t", "time"),
        (np.arange(9, dtype=float).reshape(3, 3) / 10, "xt", "xt"),
    ),
)
def test_simulation_input_previews_use_their_own_axes_and_matrix_orientation(
    field: np.ndarray, form: str, expected_kind: str,
) -> None:
    result = ConstantSimulationResult(
        position_kb=np.asarray([30.0, 32.0, 34.0]),
        replication_timing=np.asarray([4.0, 5.0, 6.0]),
        elapsed_seconds=0.1, replicates=2, geometry="linear",
        time_grid=np.arange(10.0), outputs={"replication_timing": np.asarray([4.0, 5.0, 6.0])},
    )
    request = SimpleNamespace(
        initiation_rate=field, fork_speed=1.4,
        region_start_kb=30.0, resolution_x=2.0, resolution_t=3.0,
    )
    preview = add_simulation_input_previews(result, request)
    assert "input:fork_speed" not in preview.outputs
    assert preview.input_forms["input:initiation_rate"] == form
    selected = select_result_output(preview, "input:initiation_rate")
    assert selected.axis_kind == expected_kind
    if form == "xt":
        np.testing.assert_array_equal(selected.values, field.T)
        np.testing.assert_array_equal(selected.y, [0, 3, 6])
    elif form == "t":
        np.testing.assert_array_equal(selected.values, field.reshape(-1))
        np.testing.assert_array_equal(selected.x, [0, 3, 6, 9])
    else:
        np.testing.assert_array_equal(selected.values, field)
        np.testing.assert_array_equal(selected.x, [30, 32, 34])

    options = _result_plot_options(
        preview, output_label="Initiation rate (input)", title="", series_label="Initiation rate",
        x_axis_label="", y_axis_label="", color="#b03412", palette="viridis",
        view_mode="Default", histogram_bins=10, density=False,
        show_legend=True, show_colorbar=True, log_y=False, invert_y=False, y_limits=None,
    )
    if form == "xt":
        np.testing.assert_array_equal(
            [cell[2] for cell in options["series"][-1]["data"]], field.T.reshape(-1),
        )
    else:
        np.testing.assert_array_equal(
            options["series"][0]["data"], np.column_stack((selected.x, selected.values)),
        )
