"""Plotting helpers for DNAscape outputs."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, is_color_like
from scipy import stats
from scipy.stats import gaussian_kde, mode
from scipy.ndimage import zoom

from .io import load_gene_models_from_gtf_gz
from .utils import match_yaxis_scales


def _sampled_axis_end(x):
    """Return the displayed upper edge for a regularly sampled x-axis."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(np.nanmax(x)) if x.size else 0.0

    diffs = np.diff(x)
    if np.all(diffs > 0):
        step = float(np.median(diffs))
        if step > 0 and np.allclose(diffs, step):
            return float(x[-1] + step)
    return float(np.nanmax(x))


def _apply_transparent_background(fig):
    if fig is None:
        return
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    for axis in fig.axes:
        axis.set_facecolor("none")
        axis.patch.set_alpha(0)


def plot_gene_track(
    genes,
    chrom: str,
    start: int,
    end: int,
    ax=None,
    show_names=True,
    show_direction=True,
    denseQ=False,
    x_show=True,
    label_fontsize=8,
    exon_height=0.2,
    gene_linewidth=1.0,
    max_label_chars=18,
    arrow_frac=0.015,
    arrow_min_bp=200,
    pad_bp=0
):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    ax.set_xlim(start, end)
    ax.set_yticks([])
    ax.set_ylabel("Genes")

    if x_show:
        ax.set_xlabel(f"{chrom} position (kb)")
    else:
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    span = max(1, end - start)
    arrow_len = max(arrow_min_bp, int(arrow_frac * span))

    # ----- assign rows -----
    genes_sorted = sorted(genes, key=lambda g: (g["start"], g["end"]))

    if denseQ:
        row_of = [i % 2 for i in range(len(genes_sorted))]
        nrows = 2 if genes_sorted else 1
    else:
        row_ends = []
        row_of = []
        for g in genes_sorted:
            gs, ge = max(g["start"], start), min(g["end"], end)
            placed = False
            for r, last_end in enumerate(row_ends):
                if gs >= last_end + pad_bp:
                    row_of.append(r)
                    row_ends[r] = ge
                    placed = True
                    break
            if not placed:
                row_of.append(len(row_ends))
                row_ends.append(ge)
        nrows = max(row_of) + 1 if row_of else 1

    if ax.figure is not None and ax.figure.get_size_inches()[1] < max(2.5, 0.55 * (nrows + 1)):
        ax.figure.set_size_inches(
            ax.figure.get_size_inches()[0],
            max(2.5, 0.55 * (nrows + 1)),
            forward=True
        )

    ax.set_ylim(-1, nrows)

    def y_from_row(r):
        return nrows - 1 - r

    # ----- plot -----
    label_candidates = []
    for g, r in zip(genes_sorted, row_of):
        y = y_from_row(r)
        gs, ge = max(g["start"], start), min(g["end"], end)

        ax.plot([gs, ge], [y, y], linewidth=gene_linewidth, color="black")

        for (es, ee) in g.get("exons", []):
            es2, ee2 = max(es, start), min(ee, end)
            if ee2 <= es2:
                continue
            rect = plt.Rectangle(
                (es2, y - exon_height / 2),
                ee2 - es2,
                exon_height,
                fill=True,
                facecolor="black",
                edgecolor="black",
                linewidth=0.0
            )
            ax.add_patch(rect)

        strand = g.get("strand", ".")
        if show_direction and strand in {"+", "-"}:
            if strand == "+":
                x0, x1 = max(gs, ge - arrow_len), ge
            else:
                x0, x1 = min(ge, gs + arrow_len), gs
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", lw=gene_linewidth, color="black"),
            )

        if show_names:
            name = g.get("gene_name", g["gene_id"])
            if len(name) > max_label_chars:
                name = name[: max_label_chars - 1] + "â€¦"
            label = name
            if show_direction and strand in {"+", "-"}:
                label = f"{name} ({strand})"

            xc = 0.5 * (gs + ge)
            t = ax.text(
                xc, y + 0.35, label,
                fontsize=label_fontsize,
                va="bottom",
                ha="center",
                color="black",
                clip_on=True
            )
            t.set_clip_path(ax.patch)
            t.set_visible(False)  # decide visibility after measuring
            label_candidates.append((ge - gs, r, t))

    # show non-overlapping labels, prioritising larger genes
    if show_names and label_candidates:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()

        occupied = {r: [] for r in set(rr for _, rr, _ in label_candidates)}
        pad = 0.0  # data units; keep 0 unless you want extra spacing

        for _, r, t in sorted(label_candidates, key=lambda x: x[0], reverse=True):
            bb = t.get_window_extent(renderer=renderer).transformed(inv)
            x0, x1 = bb.x0 - pad, bb.x1 + pad

            ok = True
            for a0, a1 in occupied.get(r, []):
                if not (x1 <= a0 or x0 >= a1):
                    ok = False
                    break

            if ok:
                t.set_visible(True)
                occupied.setdefault(r, []).append((x0, x1))

    ax.grid(True, axis="x", linewidth=0.5)
    return ax

def plotf(*arrays, labels=None, x_array=None, dual_axis=False, resolution=1, region_high=None, rname='Region',
          invyQ=False, figsize=(10, 4), limsx=(None, None), limsy=(None, None), title='',
          xtitle='Index', ytitle='Value', x_show=True, y_show=True, logyQ=False, logxQ=False,
          scale_matchQ=False, saveQ=False, sname='test', ext='pdf', layout_rect=(0.12, 0.15, 0.98, 0.95),
          fig=None, ax=None, showQ=True, return_handles=False, time_array=None, resolution_t=5,
          cmap='viridis', colorbarQ=True, logzQ=False, limsz=None, label_loc="lower right",
          colors=None, statsQ=False, fillupQ=False, filldownQ=False, fillintQ=False,
          transparentQ=True, **kwargs):

    def _as_plot_array(value):
        try:
            return np.asarray(value)
        except ValueError:
            return np.asarray(value, dtype=object)

    def _is_nested_array_collection(arr):
        return (
            arr.dtype == object
            and arr.ndim == 1
            and arr.size > 0
            and all(np.ndim(item) > 0 for item in arr)
        )

    def _matching_1d_axis(axis, size):
        if axis is None:
            return False
        axis_arr = np.asarray(axis)
        return axis_arr.ndim == 1 and axis_arr.size == size

    def _should_use_heatmap(arr):
        if dual_axis or arr.ndim != 2:
            return False
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            return False
        if arr.shape[0] == 1 and arr.shape[1] > 1 and x_array is not None:
            return time_array is None or _matching_1d_axis(time_array, arr.shape[1])
        if _matching_1d_axis(x_array, arr.shape[0]):
            return True
        if x_array is None and _matching_1d_axis(time_array, arr.shape[1]):
            return True
        return False

    def _normalize_colors(color_values, count, name="colors", use_first=False):
        if color_values is None:
            return [None] * count
        if is_color_like(color_values):
            return [color_values] * count
        if isinstance(color_values, np.ndarray) and color_values.ndim == 0:
            value = color_values.item()
            if value is None:
                return [None] * count
            if not is_color_like(value):
                raise ValueError(f"{name} contains an invalid color: {value!r}")
            return [value] * count

        values = list(color_values)
        if len(values) == 0:
            raise ValueError(f"{name} must contain at least one color.")
        if len(values) == 1:
            if values[0] is None:
                return [None] * count
            if not is_color_like(values[0]):
                raise ValueError(f"{name} contains an invalid color: {values[0]!r}")
            return values * count
        if len(values) != count:
            if use_first:
                if values[0] is None:
                    return [None]
                if not is_color_like(values[0]):
                    raise ValueError(f"{name} contains an invalid color: {values[0]!r}")
                return [values[0]]
            raise ValueError(f"{name} must contain one color, or one color per plotted curve.")
        for value in values:
            if value is not None and not is_color_like(value):
                raise ValueError(f"{name} contains an invalid color: {value!r}")
        return values

    def _color_kwargs(color):
        return {} if color is None else {"color": color}

    def _normalize_flags(flag_values, count, name, use_first=False):
        if flag_values is None:
            return [False] * count
        if isinstance(flag_values, (bool, np.bool_)):
            return [bool(flag_values)] * count
        if isinstance(flag_values, np.ndarray) and flag_values.ndim == 0:
            return [bool(flag_values.item())] * count
        try:
            values = list(flag_values)
        except TypeError:
            return [bool(flag_values)] * count

        if len(values) == 0:
            raise ValueError(f"{name} must contain at least one boolean.")
        if len(values) == 1:
            return [bool(values[0])] * count
        if len(values) != count:
            if use_first:
                return [bool(values[0])]
            raise ValueError(f"{name} must contain one boolean, or one boolean per plotted curve.")
        return [bool(value) for value in values]

    arrays = [_as_plot_array(a) for a in arrays]
    if len(arrays) == 1:
        arr = arrays[0]
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = _as_plot_array(arr[0])
        if arr.ndim == 2 and not _should_use_heatmap(arr):
            if arr.shape[1] == 1 and arr.shape[0] > 1:
                arrays = [arr.reshape(-1)]
            else:
                arrays = [np.asarray(row) for row in arr]
        elif _is_nested_array_collection(arr):
            arrays = [np.asarray(item) for item in arr]
        else:
            arrays = [arr]

    if x_array is not None:
        x_candidate = np.asarray(x_array)
        if x_candidate.ndim == 1 and x_candidate.size > 1:
            arrays = [
                np.full(x_candidate.size, float(arr), dtype=float)
                if np.asarray(arr).ndim == 0
                else arr
                for arr in arrays
            ]
    else:
        arrays = [
            np.asarray(arr).reshape(1)
            if np.asarray(arr).ndim == 0
            else arr
            for arr in arrays
        ]

    def _is_default_limit(value):
        return (
            value is None
            or (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and value[0] is None
                and value[1] is None
            )
        )

    if "colours" in kwargs:
        if colors is not None:
            raise ValueError("Pass only one of colors or colours.")
        colors = kwargs.pop("colours")

    for old_name, new_name in (("xlims", "limsx"), ("ylims", "limsy"), ("zlims", "limsz")):
        if old_name in kwargs:
            if new_name == "limsx":
                if not _is_default_limit(limsx):
                    raise ValueError("Pass only one of limsx or xlims.")
                limsx = kwargs.pop(old_name)
            elif new_name == "limsy":
                if not _is_default_limit(limsy):
                    raise ValueError("Pass only one of limsy or ylims.")
                limsy = kwargs.pop(old_name)
            else:
                if limsz is not None:
                    raise ValueError("Pass only one of limsz or zlims.")
                limsz = kwargs.pop(old_name)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"plotf() got unexpected keyword argument(s): {unknown}")
    limsx = (None, None) if limsx is None else limsx
    limsy = (None, None) if limsy is None else limsy
    if not isinstance(limsx, (list, tuple)) or len(limsx) != 2:
        raise ValueError("limsx must be None or a (minimum, maximum) pair.")
    if not isinstance(limsy, (list, tuple)) or len(limsy) != 2:
        raise ValueError("limsy must be None or a (minimum, maximum) pair.")
    n = len(arrays)
    statsQ = bool(statsQ)
    heatmapQ = (
        n == 1
        and arrays[0].ndim == 2
        and not (arrays[0].shape[1] == 1 and arrays[0].shape[0] > 1)
        and not dual_axis
    )
    if statsQ and heatmapQ:
        raise ValueError("statsQ=True summarizes multiple 1D curves and cannot be used with heatmap inputs.")
    if statsQ and dual_axis:
        raise ValueError("statsQ=True cannot be combined with dual_axis=True.")

    fill_count = 1 if statsQ else n
    fillup_flags = _normalize_flags(fillupQ, fill_count, "fillupQ", use_first=statsQ)
    filldown_flags = _normalize_flags(filldownQ, fill_count, "filldownQ", use_first=statsQ)
    fillintQ = bool(fillintQ)
    any_fill = any(fillup_flags) or any(filldown_flags)
    if any_fill and heatmapQ:
        raise ValueError("fillupQ/filldownQ can only be used with 1D curve plots.")
    if any_fill and dual_axis:
        raise ValueError("fillupQ/filldownQ cannot be combined with dual_axis=True.")

    line_colors = None if heatmapQ else _normalize_colors(colors, 1 if statsQ else n, use_first=statsQ)

    show_labels = True if statsQ else not (labels is None and n == 1)

    if scale_matchQ and not (dual_axis and n == 2):
        raise ValueError("scale_matchQ=True only makes sense for dual_axis with exactly two arrays.")

    if x_array is not None and not heatmapQ:
        # Allow either:
        # 1) one shared x-array for all y-arrays, or
        # 2) one x-array per y-array.
        if isinstance(x_array, np.ndarray) and x_array.ndim == 2 and x_array.shape[0] == n:
            xs = [np.asarray(row) for row in x_array]
        elif (
            isinstance(x_array, (list, tuple))
            and len(x_array) == n
            and all(np.ndim(xi) > 0 for xi in x_array)
        ):
            xs = [np.asarray(xi) for xi in x_array]
        else:
            x_shared = np.asarray(x_array)
            if x_shared.ndim != 1:
                raise ValueError(
                    "x_array must be a 1D array, or a list/tuple of 1D arrays matching number of arrays."
                )
            xs = [x_shared] * n

        for i, (xi, yi) in enumerate(zip(xs, arrays)):
            if np.asarray(xi).ndim != 1:
                raise ValueError(f"x_array[{i}] must be 1D")
            if len(xi) != len(yi):
                raise ValueError(f"x_array[{i}] and arrays[{i}] must have same length")
    elif not heatmapQ:
        xs = [np.arange(len(arr)) * resolution for arr in arrays]

    def draw_region(ax_):
        if region_high is None:
            return None
        region_high_arr = np.asarray(region_high)
        if len(region_high_arr) == 0:
            return None

        spans = []
        start = region_high_arr[0]
        prev = start
        for x in region_high_arr[1:]:
            if x == prev + resolution:
                prev = x
            else:
                spans.append((start, prev + resolution))
                start = x
                prev = x
        spans.append((start, prev + resolution))

        h0 = None
        for a, b in spans:
            h = ax_.axvspan(a, b, color="lightgrey", alpha=0.4, zorder=0)
            if h0 is None:
                h0 = h
        return h0

    def _no_offset(ax_):
        ax_.yaxis.get_offset_text().set_visible(False)

    def _apply_layout(fig_):
        fig_.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1],
                             right=layout_rect[2], top=layout_rect[3])

    def _limit_pair(lims, name):
        lims = (None, None) if lims is None else lims
        if not isinstance(lims, (list, tuple)) or len(lims) != 2:
            raise ValueError(f"{name} must be None or a (minimum, maximum) pair.")
        return lims[0], lims[1]

    def _dual_limit_pairs(lims, name):
        lims = (None, None) if lims is None else lims
        if not isinstance(lims, (list, tuple)) or len(lims) != 2:
            raise ValueError(
                f"{name} must be None, a shared (minimum, maximum) pair, "
                "or a pair of limit pairs for dual axes."
            )
        first, second = lims
        nested = (
            isinstance(first, (list, tuple))
            or isinstance(second, (list, tuple))
        )
        if nested:
            return _limit_pair(first, name), _limit_pair(second, name)
        return _limit_pair(lims, name), _limit_pair(lims, name)

    def _apply_axis_limits(ax_, limsx_=None, limsy_=None, invyQ_=False):
        if limsx_ is not None:
            ax_.set_xlim(*_limit_pair(limsx_, "limsx"))
        if limsy_ is not None:
            ax_.set_ylim(*_limit_pair(limsy_, "limsy"))
        if invyQ_:
            ax_.invert_yaxis()

    def _legend_loc(loc):
        if loc is None:
            return None
        aliases = {
            "top right": "upper right",
            "top left": "upper left",
            "bottom right": "lower right",
            "bottom left": "lower left",
            "top center": "upper center",
            "bottom center": "lower center",
            "center right": "center right",
            "center left": "center left",
        }
        key = str(loc).strip().lower().replace("_", " ").replace("-", " ")
        return aliases.get(key, loc)

    def _align_multiline_legend_handles(legend):
        if legend is None:
            return
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = getattr(legend, "legendHandles", ())
        fontsize = float(legend.prop.get_size_in_points())
        for handle, text in zip(handles, legend.get_texts()):
            label = text.get_text()
            if "\n" not in label:
                continue
            text.set_multialignment("left")
            line_count = label.count("\n") + 1
            line_spacing = float(getattr(text, "_linespacing", 1.2))
            shift = 0.5 * (line_count - 1) * fontsize * line_spacing
            if hasattr(handle, "get_ydata") and hasattr(handle, "set_ydata"):
                handle.set_ydata(np.asarray(handle.get_ydata(), dtype=float) + shift)
            elif hasattr(handle, "get_y") and hasattr(handle, "set_y"):
                handle.set_y(handle.get_y() + shift)

    def _draw_curve_fills(ax_, curve_records):
        if not any_fill or not curve_records:
            return

        xlim = ax_.get_xlim()
        ylim = ax_.get_ylim()
        y_low, y_high = sorted(ylim)
        selected = []

        for i, record in enumerate(curve_records):
            up = fillup_flags[i] if i < len(fillup_flags) else False
            down = filldown_flags[i] if i < len(filldown_flags) else False
            if not (up or down):
                continue
            x = np.asarray(record["x"], dtype=float)
            y = np.asarray(record["y"], dtype=float)
            if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
                raise ValueError("fillupQ/filldownQ require one-dimensional curves with matching x values.")
            selected.append((x, y, record["color"], up, down))

        if not selected:
            return

        alpha = 0.18
        if fillintQ:
            x0, y0, color0, up0, down0 = selected[0]
            lower = np.full_like(y0, y_low, dtype=float)
            upper = np.full_like(y0, y_high, dtype=float)
            mask = np.isfinite(x0)

            for x, y, _, up, down in selected:
                if x.shape != x0.shape or not np.allclose(x, x0, equal_nan=True):
                    raise ValueError("fillintQ=True requires all selected filled curves to share the same x values.")
                mask &= np.isfinite(y)
                if up:
                    lower = np.maximum(lower, y)
                if down:
                    upper = np.minimum(upper, y)

            mask &= lower <= upper
            if np.any(mask):
                ax_.fill_between(
                    x0,
                    lower,
                    upper,
                    where=mask,
                    interpolate=True,
                    color=color0,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
        else:
            for x, y, color, up, down in selected:
                mask = np.isfinite(x) & np.isfinite(y)
                if not np.any(mask):
                    continue
                if down:
                    ax_.fill_between(
                        x,
                        y_low,
                        y,
                        where=mask,
                        interpolate=True,
                        color=color,
                        alpha=alpha,
                        linewidth=0,
                        zorder=1,
                    )
                if up:
                    ax_.fill_between(
                        x,
                        y,
                        y_high,
                        where=mask,
                        interpolate=True,
                        color=color,
                        alpha=alpha,
                        linewidth=0,
                        zorder=1,
                    )

        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)

    legend_loc = _legend_loc(label_loc)

    created_axes = ax is None

    if heatmapQ:
        arr = arrays[0].astype(float, copy=False)

        if x_array is None:
            if arr.shape[0] == 1:
                data = arr.reshape(-1, 1)
                x_values = np.array([0.0])
            else:
                data = arr.T
                x_values = np.arange(arr.shape[0], dtype=float) * float(resolution)
        else:
            x_values = np.asarray(x_array, dtype=float)
            if x_values.ndim != 1 or x_values.size == 0:
                raise ValueError("x_array must be a non-empty one-dimensional array for heatmaps.")
            if arr.shape[0] == x_values.size and arr.shape[1] > 1:
                data = arr.T
            elif arr.shape[0] == 1:
                data = np.repeat(arr.reshape(-1, 1), x_values.size, axis=1)
            else:
                raise ValueError(
                    "For heatmaps, arrays must have shape (positions, time), or shape (1, time) "
                    "when a chromosome-position x_array is supplied."
                )

        if time_array is None:
            time_values = np.arange(data.shape[0], dtype=float) * float(resolution_t)
        else:
            time_values = np.asarray(time_array, dtype=float)
            if time_values.ndim != 1 or time_values.size != data.shape[0]:
                raise ValueError("time_array must be one-dimensional and match the heatmap time axis.")

        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            if fig is None:
                fig = ax.figure

        x_start = float(np.nanmin(x_values))
        x_end = _sampled_axis_end(x_values)
        t_start = float(np.nanmin(time_values))
        t_end = _sampled_axis_end(time_values)
        image_data = data
        norm = None
        if limsz is None:
            vmin = vmax = None
        else:
            if not isinstance(limsz, (list, tuple)) or len(limsz) != 2:
                raise ValueError("limsz must be a (minimum, maximum) pair.")
            vmin, vmax = limsz
            vmin = None if vmin is None else float(vmin)
            vmax = None if vmax is None else float(vmax)
            if vmin is not None and vmax is not None and vmax <= vmin:
                raise ValueError("limsz maximum must be greater than its minimum.")

        if logzQ:
            finite_positive = data[np.isfinite(data) & (data > 0)]
            if finite_positive.size == 0:
                raise ValueError("logzQ=True requires at least one positive heatmap value.")
            if vmin is None:
                vmin = float(np.nanmin(finite_positive))
            if vmax is None:
                vmax = float(np.nanmax(finite_positive))
            if vmin <= 0:
                raise ValueError("logzQ=True requires a positive limsz minimum.")
            if vmax <= vmin:
                vmax = vmin * 10.0
            image_data = np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)
            norm = LogNorm(vmin=vmin, vmax=vmax)

        image_options = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
        im = ax.imshow(
            image_data,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=(x_start, x_end, t_start, t_end),
            cmap=cmap,
            **image_options,
        )
        draw_region(ax)

        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        x_lower = x_start if limsx[0] is None else limsx[0]
        x_upper = x_end if limsx[1] is None else limsx[1]
        y_lower = t_start if limsy[0] is None else limsy[0]
        y_upper = t_end if limsy[1] is None else limsy[1]

        if logxQ:
            positive_x = x_values[np.isfinite(x_values) & (x_values > 0)]
            if positive_x.size == 0 or x_upper <= 0:
                raise ValueError("logxQ=True requires positive heatmap x-axis values.")
            if x_lower <= 0:
                x_lower = float(np.nanmin(positive_x))

        if logyQ:
            positive_t = time_values[np.isfinite(time_values) & (time_values > 0)]
            if positive_t.size == 0 or y_upper <= 0:
                raise ValueError("logyQ=True requires positive heatmap y-axis values.")
            if y_lower <= 0:
                y_lower = float(np.nanmin(positive_t))

        if logxQ:
            ax.set_xscale('log')
        if logyQ:
            ax.set_yscale('log')
        _apply_axis_limits(ax, (x_lower, x_upper), (y_lower, y_upper), invyQ_=invyQ)

        label = None
        if labels is not None:
            label = labels[0] if isinstance(labels, (list, tuple)) and labels else labels
        if colorbarQ:
            cax = ax.inset_axes([1.01, 0.0, 0.018, 1.0], transform=ax.transAxes)
            cbar = fig.colorbar(im, cax=cax)
            if label:
                cbar.set_label(label)

        if not x_show:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if not y_show:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        _no_offset(ax)
        ax.set_title(title)

        if transparentQ:
            _apply_transparent_background(fig)
        if created_axes:
            _apply_layout(fig)
            if saveQ:
                fig.savefig(
                    f"figures/plot_{sname}.{ext}",
                    bbox_inches="tight",
                    transparent=transparentQ,
                )
            if showQ:
                plt.show()

        if return_handles:
            return fig, ax
        return None

    if n == 2 and dual_axis:
        if ax is None:
            if fig is None:
                fig, ax1 = plt.subplots(figsize=figsize)
            else:
                ax1 = fig.add_subplot(111)
        else:
            ax1 = ax
            if fig is None:
                fig = ax1.figure

        ax2 = ax1.twinx()
        region_handle = draw_region(ax1)

        if labels is not None:
            if len(labels) != 2:
                raise ValueError("For dual_axis=True, labels must have length 2 (or be None).")
            lab1, lab2 = labels
        else:
            lab1, lab2 = ("array_1", "array_2") if show_labels else (None, None)

        color1 = line_colors[0] if line_colors[0] is not None else 'tab:blue'
        color2 = line_colors[1] if line_colors[1] is not None else 'tab:red'
        p1, = ax1.plot(xs[0], arrays[0], color=color1)
        p2, = ax2.plot(xs[1], arrays[1], color=color2)

        if isinstance(logyQ, tuple):
            logyQ1, logyQ2 = logyQ
        else:
            logyQ1 = logyQ2 = bool(logyQ)

        if logyQ1:
            ax1.set_yscale('log')
        if logyQ2:
            ax2.set_yscale('log')

        if isinstance(logxQ, tuple):
            logxQ1, logxQ2 = logxQ
        else:
            logxQ1 = logxQ2 = bool(logxQ)

        if logxQ1:
            ax1.set_xscale('log')
        if logxQ2:
            ax2.set_xscale('log')

        if isinstance(invyQ, tuple):
            invyQ1, invyQ2 = invyQ
        else:
            invyQ1 = invyQ2 = bool(invyQ)

        if scale_matchQ:
            match_yaxis_scales(ax1, ax2, arrays[0], arrays[1])

        limsy1, limsy2 = _dual_limit_pairs(limsy, "limsy")
        _apply_axis_limits(ax1, limsx, limsy1, invyQ_=invyQ1)
        _apply_axis_limits(ax2, None, limsy2, invyQ_=invyQ2)

        ax1.set_xlabel(xtitle)
        ax1.set_ylabel(lab1 if lab1 is not None else ytitle, color=color1)
        ax2.set_ylabel(lab2 if lab2 is not None else ytitle, color=color2)

        ax1.set_title(title)

        if show_labels:
            handles = []
            labels_leg = []
            if region_handle is not None:
                handles.append(region_handle)
                labels_leg.append(f"{rname}")
            if lab1 is not None:
                handles.append(p1)
                labels_leg.append(lab1)
            if lab2 is not None:
                handles.append(p2)
                labels_leg.append(lab2)
            if labels_leg:
                legend = ax2.legend(handles, labels_leg, loc=legend_loc)
                _align_multiline_legend_handles(legend)

        if not x_show:
            ax1.set_xlabel("")
            ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if not y_show:
            for a in (ax1, ax2):
                a.set_ylabel("")
                a.tick_params(axis="y", which="both", left=False, right=False,
                              labelleft=False, labelright=False)

        _no_offset(ax1)
        _no_offset(ax2)

        if transparentQ:
            _apply_transparent_background(fig)
        if created_axes:
            _apply_layout(fig)
            if saveQ:
                fig.savefig(
                    f"figures/plot_{sname}.{ext}",
                    bbox_inches="tight",
                    transparent=transparentQ,
                )
            if showQ:
                plt.show()

        if return_handles:
            return fig, (ax1, ax2)
        return None

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure

    region_handle = draw_region(ax)
    curve_records = []

    if statsQ:
        x0 = np.asarray(xs[0], dtype=float)
        if x0.ndim != 1:
            raise ValueError("statsQ=True requires one-dimensional x values.")

        y_stack = []
        for i, (xi, arr) in enumerate(zip(xs, arrays)):
            xi = np.asarray(xi, dtype=float)
            yi = np.asarray(arr, dtype=float)
            if xi.ndim != 1:
                raise ValueError(f"x_array[{i}] must be 1D when statsQ=True.")
            if yi.ndim != 1:
                raise ValueError(f"arrays[{i}] must be 1D when statsQ=True.")
            if xi.size != x0.size or yi.size != x0.size:
                raise ValueError("statsQ=True requires all curves to have the same length.")
            if not np.allclose(xi, x0, equal_nan=True):
                raise ValueError("statsQ=True requires all curves to share the same x values.")
            y_stack.append(yi)

        y_stack = np.vstack(y_stack)
        mean_y = np.nanmean(y_stack, axis=0)
        std_y = np.nanstd(y_stack, axis=0)
        summary_color = line_colors[0]
        mean_kwargs = _color_kwargs(summary_color)
        mean_line, = ax.plot(x0, mean_y, label="Mean", zorder=3, **mean_kwargs)
        summary_color = mean_line.get_color()
        curve_records.append({"x": x0, "y": mean_y, "color": summary_color})
        ax.fill_between(
            x0,
            mean_y - std_y,
            mean_y + std_y,
            color=summary_color,
            alpha=0.22,
            linewidth=0,
            label="+/- std",
            zorder=2,
        )
    else:
        for i, arr in enumerate(arrays):
            if labels is not None:
                if i >= len(labels):
                    raise ValueError("labels length must match number of arrays (or be None).")
                lab = labels[i] if show_labels else None
            else:
                lab = (f"array_{i+1}" if show_labels else None)

            line, = ax.plot(xs[i], arr, label=lab, **_color_kwargs(line_colors[i]))
            curve_records.append({"x": xs[i], "y": arr, "color": line.get_color()})

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    if logyQ:
        ax.set_yscale('log')
    if logxQ:
        ax.set_xscale('log')
    _apply_axis_limits(ax, limsx, limsy, invyQ_=invyQ)
    _draw_curve_fills(ax, curve_records)

    if show_labels:
        handles, labels_leg = ax.get_legend_handles_labels()
        if region_handle is not None:
            handles = [region_handle] + handles
            labels_leg = [f"{rname}"] + labels_leg
        if labels_leg:
            legend = ax.legend(handles, labels_leg, loc=legend_loc)
            _align_multiline_legend_handles(legend)

    if not x_show:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if not y_show:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    _no_offset(ax)

    ax.set_title(title)

    if transparentQ:
        _apply_transparent_background(fig)
    if created_axes:
        _apply_layout(fig)
        if saveQ:
            fig.savefig(
                f"figures/plot_{sname}.{ext}",
                bbox_inches="tight",
                transparent=transparentQ,
            )
        if showQ:
            plt.show()

    if return_handles:
        return fig, ax
    return None

def mplotf(marrays, mlabels=None, mx_array=None, mdual_axis=None, mresolution=None, mregion_high=None, mrname=None,
           minvyQ=None, mfigsize=None, mlimsx=None, mlimsy=None, mxtitle=None, mytitle=None,
           my_show=None, mlogyQ=None, mscale_matchQ=None, mstatsQ=None,
           mfillupQ=None, mfilldownQ=None, mfillintQ=None,
           msaveQ=None, msname=None, mext=None,
           mtime_array=None, mresolution_t=None, mcmap=None, mcolorbarQ=None, mlogzQ=None, mlimsz=None,
           mcolors=None, mtransparentQ=None,
           showgeneQ=False, gtf_gz_path="data/other/gencode/gencode.v49.annotation.gtf.gz", gene_chrom=1,
           gene_denseQ=True, gene_show_names=True, gene_show_direction=False,
           figsize=(10, 4), layout_rect=(0.12, 0.15, 0.98, 0.95), hspace=0.15,
           mheight_ratios=None, mlogxQ=None, mlabel_loc=None, **kwargs):

    k = len(marrays)
    W, H = figsize

    def at(v, i, default):
        if isinstance(v, (list, tuple)):
            return v[i] if i < len(v) else default
        return v if v is not None else default

    def at_bool(v, i, default):
        if isinstance(v, np.ndarray) and v.ndim > 0:
            return bool(v[i]) if i < len(v) else default
        return bool(at(v, i, default))

    def is_limit_pair(v):
        return (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and all(item is None or np.isscalar(item) for item in v)
        )

    def at_limits(v, i, default):
        if v is None:
            return default
        if is_limit_pair(v):
            return v
        if isinstance(v, (list, tuple)):
            return v[i] if i < len(v) else default
        return v

    def is_panel_color_collection(v):
        if v is None or is_color_like(v):
            return False
        if not isinstance(v, (list, tuple, np.ndarray)):
            return False
        values = list(v)
        if len(values) == 0:
            return False
        return all(
            item is None
            or (
                isinstance(item, (list, tuple, np.ndarray))
                and not is_color_like(item)
            )
            for item in values
        )

    def at_colors(v, i):
        if v is None:
            return None
        if is_panel_color_collection(v):
            return list(v)[i] if i < len(v) else None
        return v

    def is_panel_flag_collection(v):
        if v is None or isinstance(v, (bool, np.bool_)):
            return False
        if isinstance(v, np.ndarray):
            return v.ndim > 0
        if not isinstance(v, (list, tuple)):
            return False
        values = list(v)
        if len(values) == 0:
            return False
        return all(
            item is None
            or isinstance(item, (bool, np.bool_, list, tuple, np.ndarray))
            for item in values
        )

    def at_flags(v, i):
        if v is None:
            return None
        if is_panel_flag_collection(v):
            return list(v)[i] if i < len(v) else None
        return v

    if "mcolours" in kwargs:
        if mcolors is not None:
            raise ValueError("Pass only one of mcolors or mcolours.")
        mcolors = kwargs.pop("mcolours")
    if "fillupQ" in kwargs:
        if mfillupQ is not None:
            raise ValueError("Pass only one of mfillupQ or fillupQ.")
        mfillupQ = kwargs.pop("fillupQ")
    if "filldownQ" in kwargs:
        if mfilldownQ is not None:
            raise ValueError("Pass only one of mfilldownQ or filldownQ.")
        mfilldownQ = kwargs.pop("filldownQ")
    if "fillintQ" in kwargs:
        if mfillintQ is not None:
            raise ValueError("Pass only one of mfillintQ or fillintQ.")
        mfillintQ = kwargs.pop("fillintQ")

    for old_name, new_name in (("mxlims", "mlimsx"), ("mylims", "mlimsy"), ("mzlims", "mlimsz")):
        if old_name in kwargs:
            if new_name == "mlimsx":
                if mlimsx is not None:
                    raise ValueError("Pass only one of mlimsx or mxlims.")
                mlimsx = kwargs.pop(old_name)
            elif new_name == "mlimsy":
                if mlimsy is not None:
                    raise ValueError("Pass only one of mlimsy or mylims.")
                mlimsy = kwargs.pop(old_name)
            else:
                if mlimsz is not None:
                    raise ValueError("Pass only one of mlimsz or mzlims.")
                mlimsz = kwargs.pop(old_name)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"mplotf() got unexpected keyword argument(s): {unknown}")

    def regular_matrix(g):
        try:
            arr = np.asarray(g)
        except ValueError:
            return None
        if arr.ndim == 2 and arr.dtype != object:
            return arr
        return None

    def axis_matches(axis, size):
        if axis is None:
            return False
        axis_arr = np.asarray(axis)
        return axis_arr.ndim == 1 and axis_arr.size == size

    def first_array(g):
        return np.asarray(g[0] if isinstance(g, (list, tuple)) else g)

    def spatial_len(g, x_axis=None, time_axis=None, stats_panel=False):
        if stats_panel:
            try:
                arr = np.asarray(g)
            except ValueError:
                arr = np.asarray(g, dtype=object)
            if arr.ndim == 2 and arr.dtype != object:
                return arr.shape[1]
            first = first_array(g)
            if first.ndim == 0:
                return 1
            return len(first)

        if isinstance(g, (list, tuple)):
            mat = regular_matrix(g)
            if mat is not None:
                if axis_matches(x_axis, mat.shape[0]):
                    return mat.shape[0]
                if x_axis is None and axis_matches(time_axis, mat.shape[1]):
                    return mat.shape[0]
        arr = first_array(g)
        if arr.ndim == 2:
            if arr.shape[0] == 1 and arr.shape[1] > 1:
                return None
            if arr.shape[1] == 1 and arr.shape[0] > 1:
                return arr.shape[0]
            return arr.shape[0]
        if arr.ndim == 0:
            axis = np.asarray(x_axis) if x_axis is not None else None
            if axis is not None and axis.ndim == 1:
                return axis.size
            return 1
        return len(arr)

    x0_in = at(mx_array, 0, None)
    t0_in = at(mtime_array, 0, None)
    stats0 = at_bool(mstatsQ, 0, False)
    if x0_in is None:
        n0 = spatial_len(marrays[0], x0_in, t0_in, stats_panel=stats0) or 1
        res0 = at(mresolution, 0, 1)
        xk = np.arange(n0, dtype=float) * float(res0)  # kb
    else:
        xk = np.asarray(x0_in, float)
        n0 = spatial_len(marrays[0], x0_in, t0_in, stats_panel=stats0)
        if n0 is not None and xk.size != n0:
            raise ValueError("First mx_array length must match the length of the first plotted array.")

    xk_start = float(np.nanmin(xk))
    xk_end   = _sampled_axis_end(xk)

    gene_start_bp = int(np.floor(xk_start * 1000.0))
    gene_end_bp   = int(np.ceil(xk_end   * 1000.0))

    nrows = k + (1 if showgeneQ else 0)
    if mheight_ratios is None:
        height_ratios = None
    else:
        height_ratios = np.asarray(mheight_ratios, dtype=float)
        if height_ratios.ndim != 1 or height_ratios.size != nrows:
            raise ValueError("mheight_ratios must contain one value per subplot.")
        if not np.all(np.isfinite(height_ratios)) or np.any(height_ratios <= 0):
            raise ValueError("mheight_ratios values must be finite and greater than zero.")

    gridspec_kw = None if height_ratios is None else {"height_ratios": height_ratios}
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(W, H * nrows),
        sharex=True,
        gridspec_kw=gridspec_kw,
    )
    if nrows == 1:
        axs = [axs]

    gi = 0
    if showgeneQ:
        chrom_try = f"chr{gene_chrom}" if isinstance(gene_chrom, (int, np.integer)) else str(gene_chrom)

        genes = load_gene_models_from_gtf_gz(
            gtf_gz_path, chrom_try, gene_start_bp, gene_end_bp,
            keep_gene_types="protein_coding", max_genes=200
        )
        if len(genes) == 0:
            chrom_alt = chrom_try[3:] if chrom_try.startswith("chr") else "chr" + chrom_try
            genes = load_gene_models_from_gtf_gz(
                gtf_gz_path, chrom_alt, gene_start_bp, gene_end_bp,
                keep_gene_types="protein_coding", max_genes=200
            )
            if len(genes) > 0:
                chrom_try = chrom_alt

        genes_kb = []
        for g in genes:
            gg = dict(g)
            gg["start"] = gg["start"] / 1000.0
            gg["end"]   = gg["end"]   / 1000.0
            if "exons" in gg:
                gg["exons"] = [(es / 1000.0, ee / 1000.0) for es, ee in gg["exons"]]
            genes_kb.append(gg)

        plot_gene_track(
            genes_kb, chrom_try, xk_start, xk_end,
            ax=axs[0],
            show_names=gene_show_names,
            show_direction=gene_show_direction,
            denseQ=gene_denseQ,
            x_show=False
        )

        axs[0].set_title("")
        axs[0].grid(False)
        gi = 1

    for i in range(k):
        g = marrays[i]
        ax = axs[i + gi]
        is_list = isinstance(g, (list, tuple))
        n_in = len(g) if is_list else 1

        dualQ = bool(at(mdual_axis, i, False)) and is_list and n_in == 2
        if is_list and not dualQ and regular_matrix(g) is not None:
            args = (np.asarray(g),)
        else:
            args = tuple(g) if is_list else (g,)

        plotf(*args,
              labels=at(mlabels, i, None),
              x_array=at(mx_array, i, None),
              dual_axis=dualQ,
              resolution=at(mresolution, i, 1),
              region_high=at(mregion_high, i, None),
              rname=at(mrname, i, 'Region'),
              invyQ=at(minvyQ, i, False),
              limsx=at_limits(mlimsx, i, (None, None)),
              limsy=at_limits(mlimsy, i, (None, None)),
              xtitle=at(mxtitle, i, 'Index'),
              ytitle=at(mytitle, i, 'Value'),
              x_show=(i == k - 1),   # bottom plot only (among plotf panels)
              y_show=at(my_show, i, True),
              logxQ=at(mlogxQ, i, False),
              logyQ=at(mlogyQ, i, False),
              colors=at_colors(mcolors, i),
              scale_matchQ=at(mscale_matchQ, i, False),
              statsQ=at_bool(mstatsQ, i, False),
              fillupQ=at_flags(mfillupQ, i),
              filldownQ=at_flags(mfilldownQ, i),
              fillintQ=at_bool(mfillintQ, i, False),
              saveQ=False,
              sname=at(msname, i, 'test'),
              ext=at(mext, i, 'pdf'),
              layout_rect=layout_rect,
              fig=fig,
              ax=ax,
              showQ=False,
              time_array=at(mtime_array, i, None),
              resolution_t=at(mresolution_t, i, 5),
              cmap=at(mcmap, i, 'viridis'),
              colorbarQ=at(mcolorbarQ, i, True),
              logzQ=at(mlogzQ, i, False),
              limsz=at_limits(mlimsz, i, None),
              label_loc=at(mlabel_loc, i, "lower right"),
              transparentQ=at_bool(mtransparentQ, i, True))

    xlim_start = xk_start
    if any(bool(at(mlogxQ, i, False)) for i in range(k)) and xlim_start <= 0:
        positive_xk = xk[np.isfinite(xk) & (xk > 0)]
        if positive_xk.size == 0:
            raise ValueError("mlogxQ=True requires positive shared x-axis values.")
        xlim_start = float(np.nanmin(positive_xk))

    if mlimsx is None:
        final_limsx = (xlim_start, xk_end)
    elif is_limit_pair(mlimsx):
        final_limsx = mlimsx
    else:
        final_limsx = at_limits(mlimsx, k - 1, (xlim_start, xk_end))

    final_x_start = xlim_start if final_limsx[0] is None else final_limsx[0]
    final_x_end = xk_end if final_limsx[1] is None else final_limsx[1]
    axs[-1].set_xlim(final_x_start, final_x_end)

    # If the gene axis used set_xticks([]), it would have cleared the shared ticks.
    # Restore auto ticks + ensure bottom axis is visible.
    if showgeneQ:
        axs[-1].xaxis.set_major_locator(mticker.AutoLocator())
        axs[-1].xaxis.set_minor_locator(mticker.NullLocator())
        axs[-1].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    fig.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1],
                        right=layout_rect[2], top=layout_rect[3], hspace=hspace)

    transparentQ = any(at_bool(mtransparentQ, i, True) for i in range(k))
    if transparentQ:
        _apply_transparent_background(fig)

    if at(msaveQ, 0, False):
        fig.savefig(
            f"figures/plot_{at(msname, 0, 'test')}.{at(mext, 0, 'pdf')}",
            bbox_inches="tight",
            transparent=transparentQ,
        )

    plt.show()

def ploth(*arrays, labels=None, bin_size=5, alpha=0.5, density=False, statsQ=True,
          figsize=(10,4), xtitle='Index', ytitle='Value', xmax=None, saveQ=False, sname='test', ext='pdf'):

    arrays = [np.asarray(a) for a in arrays]
    max_val = max(np.nanmax(a) for a in arrays)

    if bin_size is None:
        bins = 50
    else:
        bins = np.arange(0, max_val + bin_size, bin_size)

    plt.figure(figsize=figsize)

    legend_entries = []

    for i, arr in enumerate(arrays):
        base_label = labels[i] if labels and i < len(labels) else None

        if statsQ:
            stats = f"μ={np.nanmean(arr):.2f}\n σ={np.nanstd(arr):.2f}\n M={np.nanmedian(arr):.2f}"
            label = (base_label if base_label else "") + stats
        else:
            label = base_label

        legend_entries.append(label)
        plt.hist(arr, bins=bins, alpha=alpha, density=density,
                 label=label)

    plt.xlabel(xtitle)
    plt.ylabel("Frequency" if not density else "Probability density")
    plt.xlim(0,xmax)

    if labels is not None or statsQ:
        leg = plt.legend(loc="upper right")
        for t in leg.get_texts():
            t.set_ha('right')

    plt.tight_layout()

    if saveQ:
        plt.savefig(f"figures/histogram_{sname}.{ext}", bbox_inches='tight')
    plt.show()

def plotb(array, label=None, resolution=1, figsize=(10,4), xtitle='Index', ytitle='Value', saveQ=False, sname='test', ext='pdf'):
    arr = np.asarray(array)
    x = np.arange(len(arr)) * resolution

    plt.figure(figsize=figsize)
    plt.bar(x, arr, width=resolution, label=label if label else "array", color='orange')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if label:
        plt.legend()

    plt.tight_layout()
    if saveQ:
        plt.savefig(f"figures/barplot_{sname}.{ext}", bbox_inches='tight')
    plt.show()

def plotd(x, y, frac=0.1, random_state=None, xlabel='X', ylabel='Y', title='Density', figsize=(10,4),
          xlim=None, ylim=None, saveQ=False, sname='test', ext='pdf'):

    # keep only valid positive values
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        print("No valid points to plot.")
        return

    # subsample
    np.random.seed(random_state)
    sample_size = int(frac * n)
    if sample_size < n:
        idx = np.random.choice(n, size=sample_size, replace=False)
        x = x[idx]
        y = y[idx]

    # KDE evaluated on a regular grid
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # determine limits if not provided
    if xlim is None:
        xmin, xmax = np.min(x), np.max(x)
    else:
        xmin, xmax = xlim

    if ylim is None:
        ymin, ymax = np.min(y), np.max(y)
    else:
        ymin, ymax = ylim

    # grid resolution
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    plt.figure(figsize=figsize)
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.yscale('log')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='Density')
    plt.tight_layout()

    if saveQ:
        plt.savefig(f"figures/dens_{sname}.{ext}", bbox_inches='tight')
    
    plt.show()

def plotw(*arrays, labels=None, figsize=(10,4), ylims=(None, None),
          xtitle='', ytitle="Value", logyQ=False, invyQ=False,
          violinQ=False, saveQ=False, sname="test", ext='pdf'):

    arrays = [np.asarray(a, float) for a in arrays]
    arrays = [a[np.isfinite(a)] for a in arrays]

    if any(a.size == 0 for a in arrays):
        raise ValueError("All input arrays must contain at least one finite value.")

    n = len(arrays)
    if labels is not None and len(labels) != n:
        raise ValueError("labels length must match number of arrays (or be None).")

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    if violinQ:
        parts = ax.violinplot(
            arrays,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.3)
            pc.set_facecolor("tab:blue")
            pc.set_edgecolor("none")

    ax.boxplot(
        arrays,
        vert=True,
        whis=1.5,
        showfliers=True,
        tick_labels=labels,
    )

    if labels is None:
        ax.set_xticks([])

    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)

    if logyQ:
        ax.set_yscale("log")
    if invyQ:
        ax.invert_yaxis()

    ax.set_ylim(ylims)

    plt.tight_layout()
    if saveQ:
        fig.savefig(f"figures/bplot_{sname}.{ext}", bbox_inches="tight")
    plt.show()

def plothm(
    field,
    resolution_hm=None,  
    smooth_type="bicubic", 
    bscale=None,
    cmap="viridis",
    x_ticks=None,
    t_ticks=None,
    xtitle="Index (x)",
    ytitle="Index (y)",
    ctitle="Value",
    figsize=(10, 4),
    xlims=(None, None),
    ylims=(None, None),
    invyQ=False,
    x_show=True,
    y_show=True,
    region_high=None,
    rname="Region",
    saveQ=False,
    sname="test",
    ext="pdf",
    layout_rect=(0.12, 0.15, 0.98, 0.95),
    fig=None,
    ax=None,
    showQ=True,
):
    # 1. Prepare Data
    F = np.asarray(field, dtype=float)
    orig_nt, orig_nx = F.shape

    # 2. Handle Resolution Tweak (Resampling)
    if resolution_hm is not None:
        zoom_t = resolution_hm / orig_nt
        zoom_x = resolution_hm / orig_nx
        F_plot = zoom(F, (zoom_t, zoom_x), order=1)
        interp_to_use = smooth_type # Use your requested smoothing
    else:
        F_plot = F
        interp_to_use = "nearest" # RAW data rendering: no interpolation

    # 3. Handle Ticks
    if x_ticks is None:
        x = np.arange(orig_nx, dtype=float)
    else:
        x = np.asarray(x_ticks, dtype=float)

    if t_ticks is None:
        t = np.arange(orig_nt, dtype=float)
    else:
        t = np.asarray(t_ticks, dtype=float)

    # 4. Setup Figure/Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) if fig is None else (fig, fig.add_subplot(111))
    else:
        fig = ax.figure if fig is None else fig

    # 5. Visibility / Limits Logic
    xmin, xmax = xlims
    ymin, ymax = ylims
    ix = np.ones(orig_nx, dtype=bool)
    it = np.ones(orig_nt, dtype=bool)
    if xmin is not None: ix &= x >= xmin
    if xmax is not None: ix &= x <= xmax
    if ymin is not None: it &= t >= ymin
    if ymax is not None: it &= t <= ymax

    F_vis = F[np.ix_(it, ix)]
    if bscale is None:
        vmin, vmax = np.nanmin(F_vis), np.nanmax(F_vis)
    else:
        vmin, vmax = bscale

    # 6. Draw Region Overlay
    def draw_region(ax_):
        if region_high is None: return None
        rh = np.sort(np.asarray(region_high, dtype=float).ravel())
        if rh.size == 0: return None
        dx = np.min(np.diff(x)) if x.size > 1 else 1.0
        spans, start, prev = [], rh[0], rh[0]
        for v in rh[1:]:
            if np.isclose(v, prev + dx):
                prev = v
            else:
                spans.append((start, prev + dx))
                start = prev = v
        spans.append((start, prev + dx))
        h0 = None
        for a, b in spans:
            h = ax_.axvspan(a, b, color="lightgrey", alpha=0.4, zorder=1)
            if h0 is None: h0 = h
        return h0

    region_handle = draw_region(ax)

    # 7. Render Heatmap
    im = ax.imshow(
        F_plot,
        aspect="auto",
        origin="lower",
        interpolation=interp_to_use, # Uses 'nearest' if resolution_hm is None
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[x[0], x[-1], t[0], t[-1]],
        zorder=0,
    )

    # 8. Formatting
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if invyQ: ax.invert_yaxis()
    if not x_show:
        ax.set_xlabel("")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if not y_show:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(ctitle)

    if region_handle is not None:
        ax.legend([region_handle], [rname], loc="upper left")

    fig.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1], 
                        right=layout_rect[2], top=layout_rect[3])

    if saveQ:
        fig.savefig(f"figures/heatmap_{sname}.{ext}", bbox_inches="tight")

    if showQ:
        plt.show()
