"""Regression tests for lazy, framework-independent simulation exports."""

from __future__ import annotations

import base64
import gzip
import io
import lzma
import re
import zipfile
import zlib
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

import app.simulation_exports as simulation_exports
from app.simulation_exports import (
    build_chart_capture_export,
    build_plot_export,
    build_raw_export,
    safe_export_filename,
    select_result_output,
)
from app.simulation_service import ConstantSimulationResult
from dnascape.io import rload
from dnascape.workflows.export_settings import get_export_settings


def _result() -> ConstantSimulationResult:
    return ConstantSimulationResult(
        position_kb=np.array([10.0, 11.0, 12.0]),
        replication_timing=np.array([18.0, 14.0, 20.0]),
        elapsed_seconds=0.1,
        replicates=2,
        geometry="linear",
        time_grid=np.array([0.0, 5.0]),
        outputs={
            "replication_timing": np.array([18.0, 14.0, 20.0]),
            "origin_efficiency": np.array([0.2, 0.6, 0.2]),
            "fork_directionality": np.array([-0.5, 0.0, 0.5]),
            "replicated_fraction": np.array([0.0, 0.75]),
            "total_forks": np.array([4.0, 7.0]),
            "completion_time": np.array([18.0, 21.0]),
            "s_phase_duration": np.array([16.0, 19.0]),
            # Pooled neighbouring-origin intervals are not one value per replicate.
            "inter_origin_distances": np.array([12.0, 8.0, 16.0, 10.0, 14.0]),
            # DNAscape's canonical detailed shape is (position, time).
            "replicated_fraction_xt": np.array(
                [
                    [0.0, 0.8],
                    [0.0, 0.7],
                    [0.0, 0.75],
                ]
            ),
            "custom_time_output": np.array([3.0, 4.0]),
        },
        detailed_outputs=True,
    )


def _png_data_uri(*, transparent: bool = False) -> tuple[str, bytes]:
    from PIL import Image

    output = io.BytesIO()
    image = Image.new("RGBA" if transparent else "RGB", (3, 2), "white")
    if transparent:
        image.putpixel((0, 0), (0, 0, 0, 0))
    image.save(output, format="PNG")
    payload = output.getvalue()
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}", payload


def _svg_data_uri() -> tuple[str, bytes]:
    payload = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" '
        'viewBox="0 0 320 180">'
        '<rect x="0" y="0" width="320" height="180" fill="#ffffff"/>'
        '<path d="M20 140 L90 90 L170 110 L290 25" fill="none" '
        'stroke="#2c7f77" stroke-width="3"/>'
        '<text x="160" y="168" text-anchor="middle" font-size="12">Position (kb)</text>'
        "</svg>"
    ).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/svg+xml;charset=UTF-8;base64,{encoded}", payload


def test_png_chart_capture_is_the_exact_preview_bitmap() -> None:
    data_uri, png_payload = _png_data_uri()
    artifact = build_chart_capture_export(
        data_uri,
        "png",
        filename="visible preview",
        dpi=150,
    )

    assert artifact.data == png_payload
    assert artifact.filename == "visible_preview.png"
    assert artifact.media_type == "image/png"


def test_svg_chart_capture_preserves_vector_paths_and_text() -> None:
    data_uri, svg_payload = _svg_data_uri()
    artifact = build_chart_capture_export(data_uri, "svg", filename="visible preview")

    assert artifact.data == svg_payload
    assert artifact.filename == "visible_preview.svg"
    assert artifact.media_type == "image/svg+xml"
    assert b"<path " in artifact.data
    assert b"<text " in artifact.data
    assert b"<image " not in artifact.data
    assert b"data:image/" not in artifact.data


def test_pdf_chart_capture_preserves_vectors_without_image_objects() -> None:
    data_uri, _ = _svg_data_uri()
    artifact = build_chart_capture_export(data_uri, "pdf", filename="visible preview")

    assert artifact.data.startswith(b"%PDF")
    assert artifact.filename == "visible_preview.pdf"
    assert artifact.media_type == "application/pdf"
    assert b"/Subtype /Image" not in artifact.data


def test_pdf_compactor_combines_safe_echarts_datum_paths() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40">'
        '<path d="M0 0l10 0l0 10l-10 0Z" fill="#a00" ecmeta_series_index="0" '
        'ecmeta_data_index="0" ecmeta_ssr_type="chart"/>'
        '<path d="M10 0l10 0l0 10l-10 0Z" fill="#00a" ecmeta_series_index="0" '
        'ecmeta_data_index="1" ecmeta_ssr_type="chart"/>'
        '<path d="M20 0l10 0l0 10l-10 0Z" fill="#a00" ecmeta_series_index="0" '
        'ecmeta_data_index="2" ecmeta_ssr_type="chart"/>'
        '<path d="M30 0l10 0l0 10l-10 0Z" fill="#a00" opacity="0.5" '
        'ecmeta_series_index="0" ecmeta_data_index="3" ecmeta_ssr_type="chart"/>'
        "</svg>"
    ).encode()

    compacted = simulation_exports._compacted_echarts_svg_for_pdf(svg)
    root = ElementTree.fromstring(compacted)
    paths = [element for element in root.iter() if element.tag.endswith("path")]

    assert len(paths) == 3
    assert paths[0].attrib["d"] == ("M0 0l10 0l0 10l-10 0Z M20 0l10 0l0 10l-10 0Z")
    assert paths[1].attrib["fill"] == "#00a"
    assert paths[2].attrib["opacity"] == "0.5"
    assert all(not key.startswith("ecmeta_") for element in root.iter() for key in element.attrib)


def test_pdf_compactor_handles_browser_svg_without_ssr_metadata() -> None:
    paths = "".join(
        f'<path d="M{index} 0l1 0l0 1l-1 0Z" fill="{"#a00" if index % 2 == 0 else "#00a"}"/>'
        for index in range(32)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40">{paths}</svg>'
    ).encode()

    compacted = simulation_exports._compacted_echarts_svg_for_pdf(svg)
    root = ElementTree.fromstring(compacted)
    paths = [element for element in root.iter() if element.tag.endswith("path")]

    assert len(paths) == 2
    assert paths[0].attrib["d"].endswith("M30 0l1 0l0 1l-1 0Z")


@pytest.mark.parametrize("file_format", ("svg", "pdf"))
def test_vector_exports_reject_canvas_png_captures(file_format: str) -> None:
    data_uri, _ = _png_data_uri()
    with pytest.raises(ValueError, match="requires a vector SVG capture"):
        build_chart_capture_export(data_uri, file_format)


@pytest.mark.parametrize("file_format", ("svg", "pdf"))
def test_vector_exports_reject_invalid_embedded_png_images(file_format: str) -> None:
    raster_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="3" height="2">'
        '<image href="data:image/png;base64,AAAA" width="3" height="2"/>'
        "</svg>"
    ).encode("utf-8")
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(raster_svg).decode("ascii")
    with pytest.raises(ValueError, match="invalid PNG image"):
        build_chart_capture_export(data_uri, file_format)


@pytest.mark.parametrize("file_format", ("svg", "pdf"))
@pytest.mark.parametrize("transparent", (False, True))
def test_heatmap_exports_embed_only_the_raster_and_keep_vector_axes(file_format: str, transparent: bool) -> None:
    image_uri, _ = _png_data_uri(transparent=transparent)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="320" height="180">'
        f'<image xlink:href="{image_uri}" x="20" y="10" width="280" height="130"/>'
        '<path d="M20 10 L20 140 L300 140" fill="none" stroke="#000000"/>'
        '<text x="160" y="168" text-anchor="middle" font-size="12">Position (kb)</text>'
        '<text x="8" y="140" font-size="10">0</text>'
        '</svg>'
    ).encode()
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")
    artifact = build_chart_capture_export(data_uri, file_format)
    if file_format == "svg":
        assert artifact.data == svg
        root = ElementTree.fromstring(artifact.data)
        assert len(list(root.iter("{http://www.w3.org/2000/svg}image"))) == 1
        assert len(list(root.iter("{http://www.w3.org/2000/svg}text"))) == 2
        assert len(list(root.iter("{http://www.w3.org/2000/svg}path"))) == 1
    else:
        assert artifact.data.startswith(b"%PDF")
        # An alpha channel is retained as a soft mask, not flattened into a
        # background that would cover the grid or misrepresent missing data.
        assert artifact.data.count(b"/Subtype /Image") == (2 if transparent else 1)
        assert (b"/SMask" in artifact.data) == transparent
        # ReportLab's content streams contain vector text/path operators, even
        # though the coloured heatmap is embedded as a separate image object.
        streams = []
        for stream in re.findall(rb"stream\r?\n(.*?)endstream", artifact.data, re.DOTALL):
            try:
                streams.append(zlib.decompress(base64.a85decode(stream.strip(), adobe=True)))
            except (ValueError, zlib.error):
                continue
        content = b"\n".join(streams)
        assert b"Position \\(kb\\)" in content
        assert b"(0) Tj" in content
        assert b"20 140 l" in content


def test_output_selection_infers_space_time_replicate_and_xt_axes() -> None:
    result = _result()

    space = select_result_output(result, "replication_timing")
    assert space.axis_kind == "space"
    np.testing.assert_array_equal(space.x, result.position_kb)

    time = select_result_output(result, "replicated_fraction")
    assert time.axis_kind == "time"
    np.testing.assert_array_equal(time.x, result.time_grid)

    replicate = select_result_output(result, "completion_time")
    assert replicate.axis_kind == "replicate"
    np.testing.assert_array_equal(replicate.x, [1.0, 2.0])

    duration = select_result_output(result, "s_phase_duration")
    assert duration.axis_kind == "replicate"
    assert duration.label == "S-phase duration"
    assert duration.y_label == "Duration (min)"

    sample = select_result_output(result, "inter_origin_distances")
    assert sample.axis_kind == "sample"
    assert sample.label == "Inter-origin distance"
    assert sample.x_label == "Interval"
    assert sample.y_label == "Distance (kb)"
    np.testing.assert_array_equal(sample.x, [1.0, 2.0, 3.0, 4.0, 5.0])

    space_time = select_result_output(result, "replicated_fraction_xt")
    assert space_time.axis_kind == "xt"
    assert space_time.values.shape == (2, 3)
    np.testing.assert_array_equal(space_time.values, result.outputs["replicated_fraction_xt"].T)
    np.testing.assert_array_equal(space_time.x, result.position_kb)
    np.testing.assert_array_equal(space_time.y, result.time_grid)


def test_automatic_y_limits_follow_the_selected_data_and_view() -> None:
    from app.pages.simulation import _automatic_y_limits, _default_axis_labels

    result = _result()
    line_limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert line_limits[0] < float(np.min(result.replication_timing))
    assert line_limits[1] > float(np.max(result.replication_timing))

    histogram_limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Histogram",
        histogram_bins=8,
        density=False,
        log_y=False,
    )
    assert histogram_limits[0] == 0.0
    assert histogram_limits[1] > 0.0

    log_limits = _automatic_y_limits(
        result,
        output_label="Replicated fraction (time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=True,
    )
    assert 0.0 < log_limits[0] < log_limits[1]

    assert _default_axis_labels(
        output_label="Replication timing",
        view_mode="Histogram",
        density=False,
    ) == ("Replication timing", "Count")
    assert _default_axis_labels(
        output_label="S-phase duration",
        view_mode="Default",
        density=True,
    ) == ("Duration (min)", "Density")


def test_output_defaults_use_scientific_domains_and_clean_dynamic_ticks() -> None:
    from app.pages.simulation import (
        _automatic_y_limits,
        _default_y_limits,
        _result_plot_options,
    )

    result = _result()
    for output_label, expected in (
        ("Fork directionality", (-1.0, 1.0)),
        ("Replicated fraction (time)", (0.0, 1.0)),
    ):
        assert _default_y_limits(output_label) == expected
        assert (
            _automatic_y_limits(
                result,
                output_label=output_label,
                view_mode="Default",
                histogram_bins=60,
                density=False,
                log_y=False,
            )
            == expected
        )

    assert _default_y_limits("Origin efficiency") is None
    efficiency_limits = _automatic_y_limits(
        result, output_label="Origin efficiency", view_mode="Default",
        histogram_bins=60, density=False, log_y=False,
    )
    assert efficiency_limits[0] == 0.0
    assert 0.6 < efficiency_limits[1] < 1.0
    assert _default_y_limits("Replication timing") is None
    automatic_limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert 0.0 < automatic_limits[0] < float(np.min(result.replication_timing))
    assert automatic_limits[1] > float(np.max(result.replication_timing))
    # Automatic bounds follow the exact data span with a small scale-relative
    # margin; readable ticks must not enlarge the visible scientific domain.
    assert automatic_limits == pytest.approx((13.76, 20.24))

    automatic_options = _result_plot_options(
        result,
        output_label="Replication timing",
        title="Replication timing",
        series_label="Mean",
        x_axis_label="Position (kb)",
        y_axis_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=automatic_limits,
    )
    assert automatic_options["yAxis"]["min"] == automatic_limits[0]
    assert automatic_options["yAxis"]["max"] == automatic_limits[1]

    manual_options = _result_plot_options(
        result,
        output_label="Replication timing",
        title="Replication timing",
        series_label="Mean",
        x_axis_label="Position (kb)",
        y_axis_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=(4.0, 12.0),
    )
    assert manual_options["yAxis"]["min"] == 4.0
    assert manual_options["yAxis"]["max"] == 12.0


def test_fixed_quantity_domains_do_not_leak_into_histogram_log_or_xt_axes() -> None:
    from app.pages.simulation import _automatic_y_limits, _default_y_limits

    result = _result()

    assert _default_y_limits("Fork directionality", view_mode="Histogram") is None
    histogram_limits = _automatic_y_limits(
        result,
        output_label="Fork directionality",
        view_mode="Histogram",
        histogram_bins=3,
        density=False,
        log_y=False,
    )
    assert histogram_limits[0] == 0.0
    assert histogram_limits[1] > 0.0
    assert histogram_limits != (-1.0, 1.0)

    assert _default_y_limits("Origin efficiency", log_y=True) is None
    log_limits = _automatic_y_limits(
        result,
        output_label="Origin efficiency",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=True,
    )
    assert 0.0 < log_limits[0] < log_limits[1]

    assert _default_y_limits("Replicated fraction (space-time)") is None
    xt_limits = _automatic_y_limits(
        result,
        output_label="Replicated fraction (space-time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert xt_limits[0] <= float(np.min(result.time_grid))
    assert xt_limits[1] == pytest.approx(float(np.max(result.time_grid)))


@pytest.mark.parametrize(
    "output_label",
    (
        "Replicated fraction (space-time)",
        "Initiation events (space-time)",
        "Coalescence events (space-time)",
        "Right fork density (space-time)",
        "Left fork density (space-time)",
    ),
)
def test_all_space_time_outputs_end_at_the_latest_mean_replication_time(
    output_label: str,
) -> None:
    from app.pages.simulation import _OUTPUT_SPECS, _automatic_y_limits

    base = _result()
    time_grid = np.arange(0.0, 35.0, 5.0)
    outputs = dict(base.outputs)
    for label in (
        "Replicated fraction (space-time)",
        "Initiation events (space-time)",
        "Coalescence events (space-time)",
        "Right fork density (space-time)",
        "Left fork density (space-time)",
    ):
        outputs[_OUTPUT_SPECS[label].key] = np.ones((base.position_kb.size, time_grid.size))
    result = replace(
        base,
        replication_timing=np.asarray([18.0, np.inf, 20.0]),
        time_grid=time_grid,
        outputs=outputs,
    )

    limits = _automatic_y_limits(
        result,
        output_label=output_label,
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )

    assert limits == pytest.approx((0.0, 20.0))


def test_space_time_mean_replication_cutoff_has_safe_fallbacks() -> None:
    from app.pages.simulation import _automatic_y_limits

    base = _result()
    unavailable = replace(
        base,
        replication_timing=np.full(base.replication_timing.shape, np.inf),
    )
    unavailable_limits = _automatic_y_limits(
        unavailable,
        output_label="Replicated fraction (space-time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert unavailable_limits == pytest.approx((0.0, 5.0))

    # An inconsistent or deliberately truncated result cannot display times
    # beyond its retained field, so it ends exactly at the last tracked time.
    truncated_limits = _automatic_y_limits(
        base,
        output_label="Replicated fraction (space-time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert truncated_limits == pytest.approx((0.0, 5.0))


def test_nonnegative_count_outputs_auto_floor_at_zero_except_on_log_axes() -> None:
    from app.pages.simulation import _automatic_y_limits

    result = _result()
    linear_limits = _automatic_y_limits(
        result,
        output_label="Total forks (time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert linear_limits[0] == 0.0
    assert linear_limits[1] > float(np.max(result.outputs["total_forks"]))

    log_limits = _automatic_y_limits(
        result,
        output_label="Total forks (time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=True,
    )
    assert 0.0 < log_limits[0] < log_limits[1]


@pytest.mark.parametrize(
    ("output_label", "view_mode", "series_type"),
    (
        ("Replication timing", "Default", "line"),
        ("S-phase duration", "Default", "custom"),
        ("Replicated fraction (space-time)", "Default", "custom"),
    ),
)
def test_default_view_resolves_by_output_and_inversion_keeps_x_axis_at_bottom(
    output_label: str,
    view_mode: str,
    series_type: str,
) -> None:
    from app.pages.simulation import _result_plot_options

    options = _result_plot_options(
        _result(),
        output_label=output_label,
        title=output_label,
        series_label="Mean",
        x_axis_label="",
        y_axis_label="",
        color="#2c7f77",
        palette="viridis",
        view_mode=view_mode,
        histogram_bins=8,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=True,
        y_limits=None,
    )

    assert options["series"][0]["type"] == series_type
    assert options["yAxis"]["inverse"] is True
    assert options["xAxis"]["position"] == "bottom"
    assert options["xAxis"]["name"] == ""
    assert options["yAxis"]["name"] == ""
    if output_label == "Replicated fraction (space-time)":
        assert options["series"][0]["silent"] is True
        assert options["tooltip"]["show"] is True
        assert options["series"][1]["silent"] is False


def test_replication_timing_line_can_group_individual_profiles_and_full_ensemble_sd() -> None:
    from app.pages.simulation import _automatic_y_limits, _result_plot_options

    base = _result()
    profiles = np.asarray(
        [
            [10.0, 14.0, 18.0],
            [14.0, 18.0, 22.0],
            [18.0, 22.0, 26.0],
        ]
    )
    mean = np.mean(profiles, axis=0)
    deviation = np.std(profiles, axis=0)
    result = replace(
        base,
        replication_timing=mean,
        replicates=3,
        outputs={**base.outputs, "replication_timing": mean},
        replication_time_profiles=profiles,
        replication_timing_std=deviation,
    )

    options = _result_plot_options(
        result,
        output_label="Replication timing",
        title="Replication timing",
        series_label="Mean",
        x_axis_label="Position (kb)",
        y_axis_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=8,
        density=False,
        show_legend=True,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        show_individual_profiles=True,
        individual_profile_count=2,
        show_standard_deviation=True,
    )

    assert [series["type"] for series in options["series"]] == [
        "custom",
        "line",
        "line",
        "line",
    ]
    assert [series["name"] for series in options["series"]] == [
        "Mean ± SD",
        "Individual simulations",
        "Individual simulations",
        "Mean",
    ]
    assert options["legend"]["data"] == [
        "Individual simulations",
        "Mean",
        "Mean ± SD",
    ]
    assert options["tooltip"]["show"] is False
    assert options["series"][1]["lineStyle"]["opacity"] == pytest.approx(0.22)

    limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Default",
        histogram_bins=8,
        density=False,
        log_y=False,
        show_individual_profiles=True,
        individual_profile_count=2,
        show_standard_deviation=True,
    )
    assert limits[0] < float(np.min(mean - deviation))
    assert limits[1] > float(np.max(mean + deviation))


def test_empty_inter_origin_distribution_has_a_specific_actionable_message() -> None:
    from app.pages.simulation import _result_plot_options

    base = _result()
    result = ConstantSimulationResult(
        position_kb=base.position_kb,
        replication_timing=base.replication_timing,
        elapsed_seconds=base.elapsed_seconds,
        replicates=base.replicates,
        geometry=base.geometry,
        time_grid=base.time_grid,
        outputs={**base.outputs, "inter_origin_distances": np.array([], dtype=float)},
        detailed_outputs=True,
    )

    with pytest.raises(ValueError, match="At least two origins must fire"):
        _result_plot_options(
            result,
            output_label="Inter-origin distance",
            title="Inter-origin distance",
            series_label="Inter-origin distance",
            x_axis_label="",
            y_axis_label="",
            color="#2c7f77",
            palette="viridis",
            view_mode="Default",
            histogram_bins=8,
            density=False,
            show_legend=True,
            show_colorbar=True,
            log_y=False,
            invert_y=False,
            y_limits=None,
        )


def test_time_profiles_offer_hover_and_keep_rounded_numeric_axes() -> None:
    from app.pages.simulation import _result_plot_options

    options = _result_plot_options(
        _result(),
        output_label="Replicated fraction (time)",
        title="Replicated fraction (time)",
        series_label="Mean",
        x_axis_label="",
        y_axis_label="",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=8,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
    )

    assert options["series"][0]["type"] == "line"
    assert options["series"][0]["silent"] is False
    assert options["tooltip"]["show"] is True
    assert ":formatter" in options["xAxis"]["axisLabel"]
    assert ":formatter" in options["yAxis"]["axisLabel"]


def test_explicit_axis_kind_resolves_unknown_output_ambiguity() -> None:
    result = _result()
    with pytest.raises(ValueError, match="unambiguous axis"):
        select_result_output(result, "custom_time_output")

    selected = select_result_output(result, "custom_time_output", axis_kind="time")
    assert selected.axis_kind == "time"
    np.testing.assert_array_equal(selected.x, result.time_grid)


@pytest.mark.parametrize(
    ("file_format", "signature", "media_type"),
    (
        ("png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ("svg", b"<?xml", "image/svg+xml"),
        ("pdf", b"%PDF", "application/pdf"),
    ),
)
def test_plot_exports_are_download_ready(
    file_format: str,
    signature: bytes,
    media_type: str,
) -> None:
    artifact = build_plot_export(
        _result(),
        "replication_timing",
        file_format,
        filename="../plots/My result.csv.gz",
        title="Constant-input simulation",
        x_label="Genomic position (kb)",
        y_label="Replication time (min)",
        color="#b03412",
        invert_y=True,
        y_limits=(10.0, 24.0),
        figure_size=(4.0, 3.0),
        dpi=100,
    )

    assert artifact.data.startswith(signature)
    assert artifact.filename == f"My_result.{file_format}"
    assert artifact.media_type == media_type


def test_xt_plot_export_accepts_palette_and_normalizes_orientation() -> None:
    artifact = build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "png",
        cmap="magma",
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert artifact.data.startswith(b"\x89PNG")
    assert len(artifact.data) > 1_000


def test_plot_export_preserves_preview_titles_labels_and_series_name() -> None:
    artifact = build_plot_export(
        _result(),
        "replication_timing",
        "svg",
        title="Preview parity title",
        series_label="Preview series",
        x_label="Preview X axis",
        y_label="Preview Y axis",
        show_legend=True,
        log_y=True,
        y_limits=(10.0, 24.0),
        figure_size=(4.0, 3.0),
        dpi=100,
    )

    svg = artifact.data.decode("utf-8")
    for visible_text in (
        "Preview parity title",
        "Preview series",
        "Preview X axis",
        "Preview Y axis",
    ):
        assert visible_text in svg


@pytest.mark.parametrize(
    ("requested", "displayed"),
    ((None, (0.0, 5.0)), ((-10.0, 20.0), (-10.0, 20.0)), ((1.0, 4.0), (1.0, 4.0)), ((6.0, 10.0), (6.0, 10.0))),
)
def test_xt_plot_export_keeps_raster_coordinates_and_requested_limits(
    monkeypatch: pytest.MonkeyPatch,
    requested: tuple[float, float] | None,
    displayed: tuple[float, float],
) -> None:
    from matplotlib.figure import Figure

    captured: list[tuple[tuple[float, float], tuple[float, float], tuple[float, ...]]] = []
    original_savefig = Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        axes = self.axes[0]
        captured.append((axes.get_xlim(), axes.get_ylim(), tuple(axes.images[0].get_extent())))
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "svg",
        y_limits=requested,
        show_colorbar=False,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    x_limits, y_limits, extent = captured[0]
    assert x_limits == (extent[0], extent[1])
    assert y_limits == displayed
    assert extent[2:] == (0.0, 5.0)


def test_xt_plot_export_rejects_reversed_limits() -> None:
    with pytest.raises(ValueError, match="greater than"):
        build_plot_export(_result(), "replicated_fraction_xt", "svg", y_limits=(10.0, 6.0))


def test_xt_plot_export_respects_preview_colorbar_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    colorbar_calls = 0
    original_colorbar = Figure.colorbar

    def recording_colorbar(self, *args, **kwargs):
        nonlocal colorbar_calls
        colorbar_calls += 1
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", recording_colorbar)
    build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "svg",
        show_colorbar=False,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert colorbar_calls == 0

    build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "svg",
        show_colorbar=True,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert colorbar_calls == 1


def test_histogram_mode_and_distribution_outputs_use_histogram_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    calls: list[tuple[int, bool]] = []
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    original_hist = Axes.hist
    original_plot = Axes.plot

    def recording_hist(self, values, *args, **kwargs):
        calls.append((kwargs["bins"], kwargs["density"]))
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", recording_hist)

    def recording_plot(self, x, y, *args, **kwargs):
        curves.append((np.asarray(x), np.asarray(y)))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", recording_plot)
    build_plot_export(
        _result(),
        "replication_timing",
        "svg",
        view_mode="Histogram",
        histogram_bins=7,
        density=True,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert calls == []
    assert len(curves) == 1 and curves[0][0].size == get_export_settings().preview_points
    assert np.isfinite(curves[0][1]).all()

    calls.clear()
    build_plot_export(
        _result(),
        "completion_time",
        "svg",
        view_mode="Default",
        histogram_bins=5,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert calls == [(5, False)]

    calls.clear()
    build_plot_export(
        _result(),
        "inter_origin_distances",
        "svg",
        view_mode="Default",
        histogram_bins=9,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert calls == [(9, False)]


def test_matplotlib_objects_are_not_imported_into_module_namespace() -> None:
    assert "matplotlib" not in simulation_exports.__dict__
    assert "Figure" not in simulation_exports.__dict__
    assert "FigureCanvasAgg" not in simulation_exports.__dict__
    assert "grid" not in build_plot_export.__code__.co_names


@pytest.mark.parametrize(
    ("file_format", "delimiter", "media_type"),
    (
        ("csv", ",", "text/csv; charset=utf-8"),
        ("tsv", "\t", "text/tab-separated-values; charset=utf-8"),
    ),
)
def test_tabular_raw_exports_follow_rdata_space_profile_layout(
    file_format: str,
    delimiter: str,
    media_type: str,
) -> None:
    artifact = build_raw_export(
        _result(),
        "replication_timing",
        file_format,
        filename="simulation result",
    )
    lines = artifact.data.decode("utf-8").splitlines()
    assert lines[0] == "replication_timing"
    assert lines[1:] == ["18", "14", "20"]
    assert artifact.filename == f"simulation_result.{file_format}"
    assert artifact.media_type == media_type


def test_txt_raw_export_is_headerless_numeric_rdata_payload() -> None:
    artifact = build_raw_export(_result(), "replication_timing", "txt")
    assert artifact.media_type == "text/plain; charset=utf-8"
    assert artifact.filename == "rsim_rt.txt"
    np.testing.assert_array_equal(
        np.loadtxt(io.BytesIO(artifact.data)),
        _result().replication_timing,
    )


def test_inter_origin_distance_raw_exports_preserve_all_pooled_samples() -> None:
    result = _result()
    artifact = build_raw_export(result, "inter_origin_distances", "tsv")
    assert artifact.filename == "rsim_iod.tsv"
    assert artifact.data.decode("utf-8").splitlines() == [
        "inter_origin_distances",
        "12",
        "8",
        "16",
        "10",
        "14",
    ]

    npz = build_raw_export(result, "inter_origin_distances", "npz")
    with np.load(io.BytesIO(npz.data), allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["values"],
            result.outputs["inter_origin_distances"],
        )
        np.testing.assert_array_equal(archive["sample"], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_xt_text_export_uses_rdata_space_by_time_layout() -> None:
    artifact = build_raw_export(_result(), "replicated_fraction_xt", "csv")
    lines = artifact.data.decode("utf-8").splitlines()
    assert lines == ["t0,t1", "0,0.8", "0,0.7", "0,0.75"]


@pytest.mark.parametrize("file_format", ("npy", "npv"))
def test_removed_numpy_vector_export_formats_are_rejected(file_format: str) -> None:
    with pytest.raises(ValueError, match="Unsupported export format"):
        build_raw_export(_result(), "replication_timing", file_format)


def test_npz_export_is_self_describing_and_uses_dnascape_axis_order() -> None:
    result = _result()
    npz = build_raw_export(result, "replicated_fraction_xt", "npz")
    with np.load(io.BytesIO(npz.data), allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["values"],
            result.outputs["replicated_fraction_xt"],
        )
        assert str(archive["data_form"]) == "spacetime"
        np.testing.assert_array_equal(archive["columns"], np.array(["t0", "t1"]))
        np.testing.assert_array_equal(archive["position_kb"], result.position_kb)
        np.testing.assert_array_equal(archive["time_min"], result.time_grid)


@pytest.mark.parametrize("file_format", ("csv", "tsv", "txt", "npz"))
def test_raw_exports_round_trip_through_rdata(
    file_format: str,
    tmp_path: Path,
) -> None:
    artifact = build_raw_export(
        _result(),
        "replication_timing",
        file_format,
        filename="replication_timing",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)
    if file_format in {"csv", "tsv"}:
        loaded = rload(path, value_column="replication_timing")
    else:
        loaded = rload(path)
    np.testing.assert_array_equal(np.asarray(loaded).reshape(-1), _result().replication_timing)


def test_space_time_npz_round_trips_with_dnascape_data_form(tmp_path: Path) -> None:
    artifact = build_raw_export(
        _result(),
        "replicated_fraction_xt",
        "npz",
        filename="replicated_fraction",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)
    loaded = rload(path, data_form="auto", return_metadata=True)
    assert loaded["data_form"] == "spacetime"
    np.testing.assert_array_equal(loaded["values"], _result().outputs["replicated_fraction_xt"])


def test_optional_gzip_wraps_raw_bytes_and_suffix() -> None:
    plain = build_raw_export(_result(), "replication_timing", "csv", filename="result")
    compressed = build_raw_export(
        _result(),
        "replication_timing",
        ".CSV",
        filename="result.csv",
        gzip_output=True,
    )
    assert gzip.decompress(compressed.data) == plain.data
    assert compressed.filename == "result.csv.gz"
    assert compressed.media_type == "application/gzip"


def test_global_compression_choices_wrap_one_named_raw_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = build_raw_export(_result(), "replication_timing", "csv", filename="result")

    zipped = build_raw_export(
        _result(),
        "replication_timing",
        "csv",
        filename="result",
        compression="zip",
    )
    assert zipped.filename == "result.zip"
    with zipfile.ZipFile(io.BytesIO(zipped.data)) as archive:
        assert archive.namelist() == ["result.csv"]
        assert archive.read("result.csv") == plain.data

    xz = build_raw_export(
        _result(),
        "replication_timing",
        "csv",
        filename="result",
        compression="xz",
    )
    assert xz.filename == "result.csv.xz"
    assert lzma.decompress(xz.data) == plain.data

    calls: list[tuple[bytes, str]] = []

    def fake_7z(data: bytes, member_name: str) -> bytes:
        calls.append((data, member_name))
        return b"7z archive"

    monkeypatch.setattr(simulation_exports, "_seven_zip_bytes", fake_7z)
    seven_zip = build_raw_export(
        _result(),
        "replication_timing",
        "csv",
        filename="result",
        compression="7z",
    )
    assert calls == [(plain.data, "result.csv")]
    assert seven_zip.filename == "result.7z"
    assert seven_zip.media_type == "application/x-7z-compressed"


def test_gzipped_npz_export_round_trips_through_rdata(tmp_path: Path) -> None:
    artifact = build_raw_export(
        _result(),
        "replicated_fraction_xt",
        "npz",
        filename="replicated_fraction",
        compression="gzip",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)

    loaded = rload(path, return_metadata=True)
    assert loaded["data_form"] == "spacetime"
    np.testing.assert_array_equal(
        loaded["values"],
        _result().outputs["replicated_fraction_xt"],
    )


@pytest.mark.parametrize(
    ("filename", "extension", "gzip_output", "expected"),
    (
        (r"../../folder\\plot final.PDF", "png", False, "plot_final.png"),
        ("CON", "csv", False, "_CON.csv"),
        ("data.npy.gz", "npz", True, "data.npz.gz"),
        ("   ", "svg", False, "rsim.svg"),
    ),
)
def test_safe_filename_removes_paths_reserved_names_and_stale_suffixes(
    filename: str,
    extension: str,
    gzip_output: bool,
    expected: str,
) -> None:
    assert safe_export_filename(filename, extension, gzip_output=gzip_output) == expected


def test_raw_export_limits_recommend_safe_binary_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_exports, "MAX_TEXT_EXPORT_VALUES", 2)
    with pytest.raises(ValueError, match="Use NPZ instead"):
        build_raw_export(_result(), "replication_timing", "csv")

    monkeypatch.setattr(simulation_exports, "MAX_BINARY_EXPORT_BYTES", 1)
    with pytest.raises(ValueError, match="Binary export would contain"):
        build_raw_export(_result(), "replication_timing", "npz")


def test_invalid_formats_plot_bounds_and_missing_outputs_are_clear() -> None:
    result = _result()
    with pytest.raises(ValueError, match="Unsupported export format"):
        build_raw_export(result, "replication_timing", "xlsx")
    with pytest.raises(ValueError, match="Unsupported export format"):
        build_plot_export(result, "replication_timing", "jpg")
    with pytest.raises(ValueError, match="pixel limit"):
        build_plot_export(
            result,
            "replication_timing",
            "png",
            figure_size=(20.0, 20.0),
            dpi=600,
        )
    with pytest.raises(ValueError, match="View mode must be"):
        build_plot_export(result, "replication_timing", "png", view_mode="Scatter")
    with pytest.raises(ValueError, match="Histogram bins"):
        build_plot_export(result, "replication_timing", "png", histogram_bins=0)
    with pytest.raises(ValueError, match="is not available"):
        build_raw_export(result, "missing", "csv")
