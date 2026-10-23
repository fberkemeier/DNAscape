"""Mapping plotting workflows with the app's exact ECharts specifications.

Plot generation is optional and independent of NiceGUI. Scientific mapping
results can be rendered to portable HTML without starting the application.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .preview import (
    FIXED_QUANTITY_Y_LIMITS,
    PreviewLine,
    arrange_histogram_bins,
    automatic_y_limits,
    build_legend_options,
    build_preview_plot_options,
    curve_dash_pattern,
    finite_values,
    has_finite_values,
    histogram_domain,
    histogram_guide_options,
    limit_curve_hover,
)


ECHARTS_SCRIPT_URL = "https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js"


def prepare_echarts_runtime(entry_path: str | Path) -> str:
    """Read the app's installed ECharts ES module as an embeddable runtime.

    NiceGUI ships ECharts as a module with one large dependency chunk. The
    mapping previews use its ordinary chart API, so its lazy ECharts-GL import
    is never invoked. Converting only the final export statement preserves the
    exact installed renderer without importing NiceGUI or fetching anything.
    """

    entry = Path(entry_path)
    source = entry.read_text(encoding="utf-8")
    exported = re.search(r"export\s*\{([^}]+)\}", source)
    imported = re.search(r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", source)
    if exported is None or imported is None:
        raise ValueError("The ECharts entry module has an unsupported format.")
    local_export = re.search(r"([\w$]+)\s+as\s+echarts\b", exported.group(1))
    if local_export is None:
        raise ValueError("The entry module does not export ECharts.")
    import_aliases = {
        local: original
        for original, local in re.findall(r"([\w$]+)\s+as\s+([\w$]+)", imported.group(1))
    }
    chunk_alias = import_aliases.get(local_export.group(1))
    chunk_path = (entry.parent / imported.group(2)).resolve()
    if chunk_alias is None or chunk_path.parent != entry.parent.resolve():
        raise ValueError("The ECharts dependency chunk is not a local module.")
    runtime = chunk_path.read_text(encoding="utf-8")
    chunk_export = re.search(r"export\s*\{([^}]+)\}\s*;?\s*(?://[^\n]*)?\s*$", runtime)
    if chunk_export is None:
        raise ValueError("The ECharts dependency module has unsupported exports.")
    original_symbol = next(
        (
            original
            for original, exported_name in re.findall(
                r"([\w$]+)\s+as\s+([\w$]+)", chunk_export.group(1)
            )
            if exported_name == chunk_alias
        ),
        None,
    )
    if original_symbol is None:
        raise ValueError("The ECharts dependency module does not export its renderer.")
    return "(() => {\n" + runtime[: chunk_export.start()] + (
        f"\n;globalThis.echarts = {original_symbol};\n}})();\n"
    )


def _html_option_values(value: Any) -> Any:
    """Make the ECharts data JSON-safe, preserving missing points as null."""

    if isinstance(value, np.ndarray):
        return _html_option_values(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _html_option_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_html_option_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_plot_html(
    specification: Mapping[str, Any],
    path: str | Path,
    *,
    title: str = "DNAscape plot",
    echarts_source: str | None = None,
    dpi: float = 150,
) -> Path:
    """Write an optional standalone HTML preview from the app's chart options.

    Passing the bundled ECharts source makes the document work offline. If
    omitted, the document loads the pinned ECharts runtime when opened in a
    browser. This function makes no network requests and imports no UI code.
    """

    destination = Path(path)
    if not np.isfinite(float(dpi)) or float(dpi) <= 0:
        raise ValueError("Plot export DPI must be a positive finite number.")
    pixel_ratio = max(1.0, min(4.0, float(dpi) / 96.0))
    aspect_ratio = specification.get("dnascapeAspectRatio")
    chart_size = "height: 360px;"
    if aspect_ratio is not None:
        aspect_ratio = float(aspect_ratio)
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError("Plot aspect ratio must be a positive finite number.")
        chart_size = f"height: auto; aspect-ratio: {aspect_ratio}; min-height: 0;"
    destination.parent.mkdir(parents=True, exist_ok=True)
    options_json = json.dumps(
        _html_option_values(specification), ensure_ascii=True, allow_nan=False
    ).replace("<", "\\u003c")
    if echarts_source is None:
        runtime_script = f'<script src="{ECHARTS_SCRIPT_URL}"></script>'
    else:
        embedded_source = re.sub(r"</script", r"<\\/script", echarts_source, flags=re.IGNORECASE)
        runtime_script = "<script>\n" + embedded_source + "\n</script>"
    safe_title = escape(str(title))
    download_name = json.dumps(destination.stem, ensure_ascii=True).replace("<", "\\u003c")
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    body {{ margin: 24px; color: #625c57; background: white; font-family: sans-serif; }}
    h1 {{ font-size: 18px; font-weight: 500; }}
    #chart {{ width: 100%; {chart_size} }}
    .exports {{ display: flex; gap: 8px; margin-top: 12px; }}
    button {{ padding: 6px 12px; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <div id="chart"></div>
  <div class="exports">
    <button id="download-png" type="button">Download PNG</button>
    <button id="download-svg" type="button">Download SVG</button>
  </div>
  {runtime_script}
  <script>
    const specification = {options_json};
    // Colon-prefixed callback keys use the same convention as the app.
    function chartOptions(value) {{
      if (Array.isArray(value)) return value.map(chartOptions);
      if (value && typeof value === 'object') {{
        return Object.fromEntries(Object.entries(value).map(([key, item]) =>
          key.startsWith(':')
            ? [key.slice(1), new Function('return (' + item + ')')()]
            : [key, chartOptions(item)]
        ));
      }}
      return value;
    }}
    const chart = echarts.init(document.getElementById('chart'), null, {{renderer: 'canvas'}});
    const options = chartOptions(specification);
    chart.setOption(options);
    window.addEventListener('resize', () => chart.resize());
    function downloadImage(url, extension) {{
      const link = document.createElement('a');
      link.href = url;
      link.download = {download_name} + '.' + extension;
      document.body.appendChild(link);
      link.click();
      link.remove();
    }}
    document.getElementById('download-png').addEventListener('click', () => {{
      downloadImage(chart.getDataURL({{
        type: 'png', pixelRatio: {pixel_ratio!r}, backgroundColor: '#fff'
      }}), 'png');
    }});
    document.getElementById('download-svg').addEventListener('click', () => {{
      const surface = document.createElement('div');
      surface.style.cssText = 'position:absolute;left:-100000px;';
      surface.style.width = chart.getWidth() + 'px';
      surface.style.height = chart.getHeight() + 'px';
      document.body.appendChild(surface);
      const vector = echarts.init(surface, null, {{renderer: 'svg'}});
      vector.setOption(options);
      downloadImage(vector.getDataURL({{type: 'svg', backgroundColor: '#fff'}}), 'svg');
      vector.dispose();
      surface.remove();
    }});
  </script>
</body>
</html>
'''
    destination.write_text(html, encoding="utf-8")
    return destination

_OUTPUT_LABELS = {
    "firing_rate": "Firing rate",
    "fork_speed": "Fork speed",
    "replication_timing": "Replication timing",
    "replication_timing_fit": "Replication timing fit",
    "fit_history": "MSE",
    "residual": "Residual",
    "replicated_fraction": "Replicated fraction",
    "replicated_fraction_xt": "Replicated fraction (space-time)",
    "fork_directionality": "Fork directionality",
    "right_fork_density_xt": "Right-moving fork density",
    "left_fork_density_xt": "Left-moving fork density",
    "fork_density_xt": "Total fork density",
}


_COMPARISON_PREFIX = "comparison:"


_FIT_DIAGNOSTIC_KEY = "fit_diagnostic"


_MSE_DIAGNOSTIC_KEY = "mse_diagnostic"


_BATCH_COLOURS = (
    "#b03412",
    "#2596be",
    "#6f5a9c",
    "#c47a1b",
    "#3f6fa3",
    "#7d8f3a",
    "#9c4d70",
    "#4e7d8d",
)


def resolved_output_key(result: Any, output_key: str) -> str:
    """Resolve a virtual diagnostic key without failing UI synchronization."""

    if output_key == _MSE_DIAGNOSTIC_KEY:
        outputs = getattr(result, "outputs", {})
        return "fit_history" if "fit_history" in outputs else output_key
    if output_key != _FIT_DIAGNOSTIC_KEY:
        return output_key
    outputs = getattr(result, "outputs", {})
    return "residual" if "residual" in outputs else output_key


def _result_axis(result: Any, *, kind: str, length: int) -> np.ndarray:
    """Return a matching result axis while tolerating scalar and legacy results."""

    names = (
        ("time_min", "time_axis", "time_grid")
        if kind == "time"
        else ("position_kb", "x_axis", "position")
    )
    for name in names:
        candidate = getattr(result, name, None)
        if candidate is None:
            continue
        axis = np.asarray(candidate, dtype=float).reshape(-1)
        if axis.size == length:
            return axis
    return np.arange(length, dtype=float)


def mapping_plot_options(
    result: Any,
    *,
    output_key: str,
    plot_type: str,
    normalization: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    palette: str,
    histogram_bins: int,
    line_width: float,
    show_markers: bool,
    show_colorbar: bool,
    log_y: bool,
    invert_y: bool,
    log_z: bool = False,
    series_label: str = "",
    density: bool = False,
    show_legend: bool = True,
    legend_position: str = "Top right",
    show_grid: bool = True,
    y_limits: tuple[float, float] | None = None,
    secondary_color: str = "#2596be",
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    kde_bandwidth: float = 1.0,
    histogram_grouped: bool = False,
    histogram_boundary: str = "left",
    x_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
    aspect_ratio: float | None = None,
    line_type: str = "solid",
) -> dict[str, Any]:
    """Build a bounded ECharts view for scalar, X, T, XT, and bundle outputs.

    Observed profiles always display only their supplied bins. A smooth
    periodic firing-rate fit also shows its model value at the period end;
    this closes the model curve without changing observations or raw arrays.
    """

    output_key = resolved_output_key(result, output_key)
    from .preview_inputs import INPUT_PREFIX, input_preview_result, input_profile_label
    result = input_preview_result(result, output_key)

    if output_key.startswith(_COMPARISON_PREFIX):
        fitted_key = output_key.removeprefix(_COMPARISON_PREFIX)
        comparison = getattr(result, "comparisons", {}).get(fitted_key)
        if comparison is None:
            raise ValueError("The selected fit comparison was not returned by DNAscape.")
        if plot_type != "Line":
            raise ValueError("Observed-versus-fitted comparisons use a line plot.")
        observed = np.asarray(comparison.observed, dtype=float).reshape(-1)
        fitted = np.asarray(comparison.fitted, dtype=float).reshape(-1)
        finite = np.concatenate((finite_values(observed), finite_values(fitted)))
        if finite.size == 0:
            raise ValueError("The fit comparison contains no finite values.")
        if normalization == "Z-score":
            center = float(np.mean(finite))
            scale = float(np.std(finite))
            observed = observed - center
            fitted = fitted - center
            if scale > 0:
                observed = observed / scale
                fitted = fitted / scale
        elif normalization == "Min-max [0,1]":
            lower = float(np.min(finite))
            span = float(np.max(finite) - lower)
            observed = observed - lower
            fitted = fitted - lower
            if span > 0:
                observed = observed / span
                fitted = fitted / span
        axis_kind = "time" if comparison.form == "t" else "space"
        axis = (
            np.arange(observed.size, dtype=float)
            if comparison.form in {"scalar", "regions", "supplied"}
            else _result_axis(result, kind=axis_kind, length=observed.size)
        )
        return build_preview_plot_options(
            plot_type="Line",
            aspect_ratio=aspect_ratio,
            title=title.strip() or "Observed vs fitted",
            x_limits=x_limits,
            x_label=x_label,
            y_label=y_label,
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=show_legend,
            show_colorbar=show_colorbar,
            log_y=log_y,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=series_label,
            lines=(
                PreviewLine("Observed", axis, observed, color),
                PreviewLine("Fitted", axis, fitted, secondary_color),
            ),
            line_width=line_width,
            line_type=line_type,
            show_markers=show_markers,
            legend_position=legend_position,
            show_grid=show_grid,
            interactive=True,
            log_x=log_x,
            invert_x=invert_x,
        )

    outputs = getattr(result, "outputs", {})
    if output_key not in outputs:
        raise ValueError("The selected mapping output was not returned by DNAscape.")
    values = np.asarray(outputs[output_key])
    if values.size == 0 and output_key == "inter_origin_distance":
        return {
            "title": {"text": title.strip() or "Inter-origin distance"},
            "xAxis": {"type": "value", "name": x_label},
            "yAxis": {"type": "value", "name": y_label},
            "series": [],
            "graphic": [
                {
                    "type": "text", "left": "center", "top": "middle",
                    "style": {"text": "No pairs of fired origins were observed.", "fill": "#777"},
                }
            ],
        }
    has_finite = has_finite_values(values)
    if values.size and not has_finite and getattr(result, "diagnostics", {}).get(
        "unidentifiable_count"
    ) == values.size:
        return {
            "title": {"text": title.strip() or _OUTPUT_LABELS.get(output_key, output_key)},
            "xAxis": {"type": "value", "name": x_label},
            "yAxis": {"type": "value", "name": y_label},
            "series": [],
            "graphic": [{
                "type": "text", "left": "center", "top": "middle",
                "style": {"text": "No identifiable values in this interval.", "fill": "#777"},
            }],
        }
    if values.size == 0 or not has_finite:
        raise ValueError("The selected mapping output contains no finite values.")
    if normalization == "Z-score":
        finite = finite_values(values)
        scale = float(np.std(finite))
        values = values - float(np.mean(finite))
        if scale > 0:
            values = values / scale
    elif normalization == "Min-max [0,1]":
        finite = finite_values(values)
        lower = float(np.min(finite))
        span = float(np.max(finite) - lower)
        values = values - lower
        if span > 0:
            values = values / span

    resolved_title = title.strip() or (
        input_profile_label(output_key) if output_key.startswith(INPUT_PREFIX)
        else _OUTPUT_LABELS.get(output_key, output_key.replace("_", " ").title())
    )
    if plot_type == "Heatmap":
        output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
        if output_form == "t":
            time_axis = _result_axis(result, kind="time", length=values.size)
            space_axis = np.asarray(getattr(result, "x_axis", []), dtype=float).reshape(-1)
            if space_axis.size == 0:
                space_axis = np.asarray([0.0])
            display_matrix = np.broadcast_to(values.reshape(-1, 1), (values.size, space_axis.size))
        elif output_form == "xt":
            matrix = np.asarray(values)
            if matrix.ndim != 2:
                raise ValueError("Heatmap view requires a temporal or space-time mapping output.")
            space_axis = _result_axis(result, kind="space", length=matrix.shape[0])
            time_axis = _result_axis(result, kind="time", length=matrix.shape[1])
            # rmap returns canonical (position, time) arrays. DNAscape's
            # plotting API transposes them for a position-X, time-Y image.
            display_matrix = matrix.T
        else:
            raise ValueError("Heatmap view requires a temporal or space-time mapping output.")
        return build_preview_plot_options(
            plot_type="Heatmap",
            aspect_ratio=aspect_ratio,
            x_limits=x_limits,
            z_limits=z_limits,
            title=resolved_title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=show_legend,
            show_colorbar=show_colorbar,
            log_y=log_y,
            log_z=log_z,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=resolved_title,
            values=display_matrix,
            x_axis=space_axis,
            y_axis=time_axis,
            show_grid=show_grid,
            interactive=True,
            log_x=log_x,
            invert_x=invert_x,
            invert_z=invert_z,
        )

    flattened = values.reshape(-1)
    if plot_type == "Histogram":
        return build_preview_plot_options(
            plot_type="Histogram",
            aspect_ratio=aspect_ratio,
            x_limits=x_limits,
            title=resolved_title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=show_legend,
            show_colorbar=show_colorbar,
            log_y=log_y,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=series_label.strip() or resolved_title,
            values=flattened,
            legend_position=legend_position,
            show_grid=show_grid,
            log_x=log_x,
            invert_x=invert_x,
            show_histogram_statistics=show_histogram_statistics,
            histogram_statistics_position=histogram_statistics_position,
            histogram_bin_size=histogram_bin_size,
            histogram_x_limits=histogram_x_limits,
            histogram_touching=histogram_touching,
            kde_bandwidth=kde_bandwidth,
            histogram_grouped=histogram_grouped,
            histogram_boundary=histogram_boundary,
        )

    if plot_type == "Boxplot":
        return build_preview_plot_options(
            plot_type="Boxplot",
            aspect_ratio=aspect_ratio,
            x_limits=x_limits,
            title=resolved_title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=False,
            show_colorbar=show_colorbar,
            log_y=log_y,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=series_label.strip() or resolved_title,
            values=flattened,
            legend_position=legend_position,
            show_grid=show_grid,
            invert_x=invert_x,
        )

    if values.ndim > 1 and 1 not in values.shape:
        raise ValueError("Choose Heatmap to view this two-dimensional mapping output.")
    dependence = str(getattr(result, "output_dependence", "space"))
    output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
    axis_kind = "time" if output_form == "t" or dependence == "time" else "space"
    if output_key == "fit_history":
        axis = np.arange(1, flattened.size + 1, dtype=float)
    elif dependence in {"ensemble", "regions", "constant"}:
        axis = np.arange(flattened.size, dtype=float)
    else:
        axis = _result_axis(result, kind=axis_kind, length=flattened.size)
    if (
        output_key == "firing_rate" and output_form == "x" and axis.size > 1
        and bool(getattr(result, "assumptions", {}).get("perQ", False))
        and float(getattr(result, "diagnostics", {}).get("fit_periodic_smoothing_kb", 0.)) > 0
    ):
        axis = np.append(axis, axis[-1] + (axis[-1] - axis[-2]))
        flattened = np.append(flattened, flattened[0])
    return build_preview_plot_options(
        plot_type="Line",
        aspect_ratio=aspect_ratio,
        x_limits=x_limits,
        title=resolved_title,
        x_label=x_label,
        y_label=y_label,
        color=color,
        palette=palette,
        histogram_bins=histogram_bins,
        density=density,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        log_y=log_y,
        invert_y=invert_y,
        y_limits=y_limits,
        series_label=series_label.strip() or resolved_title,
        values=flattened,
        x_axis=axis,
        line_width=line_width,
        line_type=line_type,
        show_markers=show_markers or (output_key == "fit_history" and flattened.size == 1),
        legend_position=legend_position,
        show_grid=show_grid,
        interactive=True,
        log_x=log_x,
        invert_x=invert_x,
    )


def mapping_automatic_y_limits(
    result: Any,
    *,
    output_key: str,
    plot_type: str,
    histogram_bins: int,
    density: bool,
    kde_bandwidth: float = 1.0,
    log_y: bool,
    log_x: bool = False,
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_boundary: str = "left",
) -> tuple[float, float]:
    """Return editable, data-derived bounds for the active mapping plot."""

    output_key = resolved_output_key(result, output_key)
    from .preview_inputs import input_preview_result
    result = input_preview_result(result, output_key)

    if output_key.startswith(_COMPARISON_PREFIX):
        comparison = getattr(result, "comparisons", {}).get(
            output_key.removeprefix(_COMPARISON_PREFIX)
        )
        if comparison is None:
            raise ValueError("The selected fit comparison was not returned by DNAscape.")
        values = [comparison.observed, comparison.fitted]
        y_axis = None
        fixed_limits = None
        quantity_key = comparison.source_key.removesuffix("_xt")
    else:
        output = np.asarray(getattr(result, "outputs", {}).get(output_key), dtype=float)
        output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
        if output_form == "t":
            time_length = output.size
        elif output_form == "xt" and output.ndim == 2:
            time_length = output.shape[1]
        else:
            time_length = 0
        if plot_type == "Heatmap" and time_length == 0:
            raise ValueError("Heatmap view requires a temporal or space-time output.")
        values = [output]
        y_axis = (
            _result_axis(result, kind="time", length=time_length)
            if plot_type == "Heatmap"
            else None
        )
        quantity_key = output_key.removesuffix("_xt")
        fixed_limits = FIXED_QUANTITY_Y_LIMITS.get(quantity_key)
    return automatic_y_limits(
        values,
        plot_type=plot_type,
        histogram_bins=histogram_bins,
        density=density,
        kde_bandwidth=kde_bandwidth,
        log_y=log_y,
        log_x=log_x,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_boundary=histogram_boundary,
        y_axis=y_axis,
        fixed_limits=fixed_limits,
        # A nonnegative physical quantity does not imply that zero belongs in
        # every preview.  Keep ordinary mapping profiles centred on their
        # observed extrema; genuinely bounded quantities above use their
        # explicit scientific domains, and histograms anchor their own count
        # baseline inside ``automatic_y_limits``.
        floor_at_zero=False,
    )


def combined_mapping_plot_options(
    results: Sequence[Any],
    *,
    colors: Mapping[tuple[str, str, int], str] | None = None,
    line_types: Mapping[tuple[str, str, int], str] | None = None,
    curve_names: Mapping[tuple[str, str, int], str] | None = None,
    display_names: Mapping[str, str] | None = None,
    plot_options_builder: Callable[..., dict[str, Any]] | None = None,
    **plot_kwargs: Any,
) -> dict[str, Any]:
    """Merge compatible one-dimensional Mapping previews into one EChart."""

    if len(results) < 2:
        raise ValueError("A combined Mapping preview requires several results.")
    build_options = plot_options_builder or mapping_plot_options
    plot_type = str(plot_kwargs.get("plot_type") or "Line")
    histogram = plot_type == "Histogram"
    combined_series: list[dict[str, Any]] = []
    combined_names: list[str] = []
    x_minima: list[float] = []
    x_maxima: list[float] = []
    base: dict[str, Any] | None = None
    output_key = str(plot_kwargs.get("output_key") or "output")
    if histogram and not plot_kwargs.get("density") and plot_kwargs.get("histogram_x_limits") is None:
        extents = []
        for named in results:
            values = finite_values(named.result.outputs[resolved_output_key(named.result, output_key)])
            if plot_kwargs.get("log_x"):
                values = values[values > 0]
            if values.size:
                extents.extend((float(values.min()), float(values.max())))
        domain = histogram_domain(extents, log_x=bool(plot_kwargs.get("log_x")))
        if domain is not None:
            plot_kwargs["histogram_x_limits"] = domain[1]
    axis_defaults: dict[str, list[list[float]]] = {}
    y_ranges: list[tuple[float, float]] = []
    curve_index = 0
    for named in results:
        color = _BATCH_COLOURS[curve_index % len(_BATCH_COLOURS)]
        kwargs = dict(plot_kwargs)
        kwargs["color"] = color
        kwargs["secondary_color"] = _BATCH_COLOURS[
            (curve_index + 1) % len(_BATCH_COLOURS)
        ]
        options = build_options(named.result, **kwargs)
        for key, bounds in options.get("dnascapeDefaultAxes", {}).items():
            axis_defaults.setdefault(key, []).append(list(bounds))
        y_axis = options.get("yAxis", {})
        if isinstance(y_axis, dict) and "min" in y_axis and "max" in y_axis:
            y_ranges.append((y_axis["min"], y_axis["max"]))
        series = list(options.get("series", ()))
        accepted_series = {"custom"} if histogram and not bool(plot_kwargs.get("density")) else {"line", "scatter"}
        if not series or any(item.get("type") not in accepted_series for item in series):
            raise ValueError(
                "This output cannot be combined. Choose Individual to view each dataset."
            )
        if base is None:
            base = options
        x_axis = options.get("xAxis", {})
        try:
            x_minima.append(float(x_axis["min"]))
            x_maxima.append(float(x_axis["max"]))
        except (KeyError, TypeError, ValueError):
            pass
        for series_index, item in enumerate(series):
            curve_color = (colors or {}).get(
                (named.key, output_key, series_index),
                _BATCH_COLOURS[curve_index % len(_BATCH_COLOURS)],
            )
            line_style = dict(item.get("lineStyle", {}))
            line_style["color"] = curve_color
            if not histogram or bool(plot_kwargs.get("density")):
                line_style["type"] = curve_dash_pattern((line_types or {}).get(
                    (named.key, output_key, series_index), str(plot_kwargs.get("line_type") or "solid"),
                ))
            item["lineStyle"] = line_style
            item_style = dict(item.get("itemStyle", {}))
            item_style["color"] = curve_color
            if histogram and not bool(plot_kwargs.get("density")):
                # Semi-transparent bars preserve every distribution when
                # their shared bins occupy the same coordinates.
                item_style["opacity"] = 0.48
            item["itemStyle"] = item_style
            original_name = str(item.get("name") or "Output")
            result_name = (display_names or {}).get(named.key, named.name)
            display_name = (
                result_name
                if len(series) == 1
                else f"{result_name} · {original_name}"
            )
            display_name = (curve_names or {}).get((named.key, output_key, series_index), display_name)
            item["name"] = display_name
            combined_names.append(display_name)
            combined_series.append(item)
            curve_index += 1
    assert base is not None
    base["series"] = combined_series
    base["dnascapeDefaultAxes"] = {
        key: [min(bounds[0] for bounds in ranges), max(bounds[1] for bounds in ranges)]
        for key, ranges in axis_defaults.items()
    }
    if y_ranges and plot_kwargs.get("y_limits") is None:
        base["yAxis"] = {**base["yAxis"], "min": min(bounds[0] for bounds in y_ranges),
                        "max": max(bounds[1] for bounds in y_ranges)}
    if x_minima and x_maxima:
        base_x_axis = dict(base.get("xAxis", {}))
        base_x_axis["min"] = min(x_minima)
        base_x_axis["max"] = max(x_maxima)
        base["xAxis"] = base_x_axis
    if histogram:
        datasets = []
        for index, named in enumerate(results):
            actual = resolved_output_key(named.result, output_key)
            series = combined_series[index]
            color = series.get("itemStyle", {}).get("color") or series.get("lineStyle", {}).get("color") or _BATCH_COLOURS[index % len(_BATCH_COLOURS)]
            series["dnascapeDataset"] = index
            values = finite_values(named.result.outputs[actual])
            if plot_kwargs.get("log_x"):
                values = values[values > 0]
            domain = plot_kwargs.get("histogram_x_limits")
            if domain is not None and not plot_kwargs.get("density"):
                values = values[(values >= domain[0]) & (values <= domain[1])]
            datasets.append((combined_names[index], color, values))
        histogram_guide_options(base, datasets, legend=bool(plot_kwargs.get("show_legend", True)),
                                statistics=bool(plot_kwargs.get("show_histogram_statistics", False)),
                                position=str(plot_kwargs.get("legend_position") or "Top right"))
        if plot_kwargs.get("histogram_grouped") and not plot_kwargs.get("density"):
            arrange_histogram_bins(base, len(datasets))
    else:
        legend = dict(base.get("legend", {}))
        legend["data"] = combined_names
        base["legend"] = legend
    return limit_curve_hover(base)
