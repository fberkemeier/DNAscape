"""Chromosome-sized heatmaps retain full exports and bounded previews."""

import io

import numpy as np
import pytest

from dnascape.workflows import exports, mapping_exports
from dnascape.workflows.mapping import MappingResult, _validate_grid
from dnascape.workflows.preview import build_preview_plot_options
from dnascape.workflows.simulation import (
    ConstantSimulationRequest, ConstantSimulationResult, validate_constant_simulation,
)
from dnascape.workflows.uploads import inspect_upload, materialize_upload


def test_chromosome_at_five_minute_sampling_passes_interactive_grid_guards():
    points, times = 249_251, 61
    _validate_grid({"n_bins": points, "max_time": 300., "resolution_time": 5.})
    validate_constant_simulation(ConstantSimulationRequest(
        initiation_rate=.001, fork_speed=1.4, resolution_x=1., resolution_t=5.,
        region_start_kb=0., region_end_kb=float(points), replicates=1,
        geometry="linear", time_window=300., max_replication_time=1200.,
        track_time_outputs=True, track_detailed_outputs=True,
    ))
    values = np.broadcast_to(np.linspace(0., 1., times), (points, times))
    chart = build_preview_plot_options(
        plot_type="Heatmap", title="", x_label="Position (kb)", y_label="Time (min)",
        color="#b03412", palette="viridis", histogram_bins=20, density=False,
        show_legend=True, show_colorbar=True, log_y=False, invert_y=False,
        y_limits=None, values=values, x_axis=np.arange(points), y_axis=np.arange(times)*5.,
        series_label="Replicated fraction",
    )
    assert chart["xAxis"]["max"] == points-1
    assert chart["yAxis"]["max"] == 300.
    assert not chart["xAxis"]["splitLine"]["show"]
    assert not chart["yAxis"]["splitLine"]["show"]
    assert len(chart["series"][0]["data"]) == 1  # Bounded raster, full source retained.


def test_chromosome_heatmap_can_be_reopened_as_npz():
    values = np.broadcast_to(np.linspace(0., 1., 61, dtype=np.float32), (249_251, 61))
    buffer = io.BytesIO()
    np.savez(buffer, values=values, data_form=np.array("spacetime"))
    inspected = inspect_upload("heatmap.npz", buffer.getvalue(), expected_form="xt")
    restored = materialize_upload(inspected, array_key="values")
    assert restored.shape == values.shape
    np.testing.assert_array_equal(restored[[0, -1]], values[[0, -1]])


@pytest.mark.parametrize("kind", ["mapping", "simulation"])
def test_text_export_keeps_more_than_two_million_heatmap_values(kind):
    # 40,000 rows x 53 samples reproduces the reported 2,120,000-value export.
    values = np.broadcast_to(np.ones(53), (40_000, 53))
    positions, times = np.arange(40_000.), np.arange(53.)*5.
    if kind == "mapping":
        result = MappingResult(
            target="initiation_rate", target_key="initiation_rate", target_label="Initiation rate",
            method="test", output_form="xt", output_dependence="spacetime",
            primary_output="initiation_rate", outputs={"initiation_rate": values},
            output_forms={"initiation_rate": "xt"}, x_axis=positions, time_axis=times,
            region_labels=(), elapsed_seconds=0., assumptions={},
        )
        artifact = mapping_exports.build_mapping_raw_export(result, "initiation_rate", "tsv")
    else:
        result = ConstantSimulationResult(
            position_kb=positions, replication_timing=np.ones(len(positions)),
            elapsed_seconds=0., replicates=1, geometry="linear", time_grid=times,
            outputs={"replicated_fraction_xt": values},
        )
        artifact = exports.build_raw_export(result, "replicated_fraction_xt", "tsv")
    assert artifact.data.count(b"\n") == len(positions)+1
    assert artifact.data.split(b"\n", 2)[1].split(b"\t") == [b"1"]*53


@pytest.mark.parametrize("module", [exports, mapping_exports])
def test_text_serializers_enforce_the_byte_budget(monkeypatch, module):
    monkeypatch.setattr(module, "MAX_TEXT_EXPORT_BYTES", 8)
    if module is mapping_exports:
        selected = mapping_exports.SelectedMappingOutput(
            key="timing", values=np.arange(20.), data_form="space", semantic_form="x",
            position_kb=np.arange(20.), time_min=np.array([]), region_labels=(),
        )
    else:
        selected = exports.SelectedOutput(
            key="replication_timing", values=np.arange(20.), axis_kind="space",
            x=np.arange(20.), y=None, label="Timing", x_label="Position", y_label="Timing",
        )
    with pytest.raises(ValueError, match="Text export exceeds"):
        module._text_bytes(selected, "tsv")
