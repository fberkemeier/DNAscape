"""Plotting helpers for DNAscape outputs."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
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
          invyQ=False, figsize=(10, 4), xlims=(None, None), ylims=(None, None), title='',
          xtitle='Index', ytitle='Value', x_show=True, y_show=True, logyQ=False, logxQ=False,
          scale_matchQ=False, saveQ=False, sname='test', ext='pdf', layout_rect=(0.12, 0.15, 0.98, 0.95),
          fig=None, ax=None, showQ=True, return_handles=False):

    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays)

    show_labels = not (labels is None and n == 1)

    if scale_matchQ and not (dual_axis and n == 2):
        raise ValueError("scale_matchQ=True only makes sense for dual_axis with exactly two arrays.")

    if x_array is not None:
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
    else:
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

    created_axes = ax is None

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

        p1, = ax1.plot(xs[0], arrays[0], color='tab:blue')
        p2, = ax2.plot(xs[1], arrays[1], color='tab:red')

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

        if invyQ1:
            ax1.invert_yaxis()
        if invyQ2:
            ax2.invert_yaxis()

        if scale_matchQ:
            match_yaxis_scales(ax1, ax2, arrays[0], arrays[1])

        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims if not isinstance(ylims, tuple) else ylims[0])
        ax2.set_ylim(ylims if not isinstance(ylims, tuple) else ylims[1])

        ax1.set_xlabel(xtitle)
        ax1.set_ylabel(lab1 if lab1 is not None else ytitle, color='tab:blue')
        ax2.set_ylabel(lab2 if lab2 is not None else ytitle, color='tab:red')

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
                ax2.legend(handles, labels_leg, loc="upper left")

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

        if created_axes:
            _apply_layout(fig)
            if saveQ:
                fig.savefig(f"figures/plot_{sname}.{ext}", bbox_inches="tight")
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

    for i, arr in enumerate(arrays):
        if labels is not None:
            if i >= len(labels):
                raise ValueError("labels length must match number of arrays (or be None).")
            lab = labels[i] if show_labels else None
        else:
            lab = (f"array_{i+1}" if show_labels else None)

        ax.plot(xs[i], arr, label=lab)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    if logyQ:
        ax.set_yscale('log')
    if logxQ:
        ax.set_xscale('log')
    if invyQ:
        ax.invert_yaxis()

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    if show_labels:
        handles, labels_leg = ax.get_legend_handles_labels()
        if region_handle is not None:
            handles = [region_handle] + handles
            labels_leg = [f"{rname}"] + labels_leg
        if labels_leg:
            ax.legend(handles, labels_leg)

    if not x_show:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if not y_show:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    _no_offset(ax)

    ax.set_title(title)

    if created_axes:
        _apply_layout(fig)
        if saveQ:
            fig.savefig(f"figures/plot_{sname}.{ext}", bbox_inches="tight")
        if showQ:
            plt.show()

    if return_handles:
        return fig, ax
    return None

def mplotf(marrays, mlabels=None, mx_array=None, mdual_axis=None, mresolution=None, mregion_high=None, mrname=None,
           minvyQ=None, mfigsize=None, mxlims=None, mylims=None, mxtitle=None, mytitle=None,
           my_show=None, mlogyQ=None, mscale_matchQ=None, msaveQ=None, msname=None, mext=None,
           showgeneQ=False, gtf_gz_path="data/other/gencode/gencode.v49.annotation.gtf.gz", gene_chrom=1,
           gene_denseQ=True, gene_show_names=True, gene_show_direction=False,
           figsize=(10, 4), layout_rect=(0.12, 0.15, 0.98, 0.95), hspace=0.15):

    k = len(marrays)
    W, H = figsize

    def at(v, i, default):
        if isinstance(v, (list, tuple)):
            return v[i] if i < len(v) else default
        return v if v is not None else default

    def first_len():
        g0 = marrays[0]
        return len(g0[0]) if isinstance(g0, (list, tuple)) else len(g0)

    n0 = first_len()
    x0_in = at(mx_array, 0, None)
    if x0_in is None:
        res0 = at(mresolution, 0, 1)
        xk = np.arange(n0, dtype=float) * float(res0)  # kb
    else:
        xk = np.asarray(x0_in, float)
        if xk.size != n0:
            raise ValueError("First mx_array length must match the length of the first plotted array.")

    xk_start = float(np.nanmin(xk))
    xk_end   = _sampled_axis_end(xk)

    gene_start_bp = int(np.floor(xk_start * 1000.0))
    gene_end_bp   = int(np.ceil(xk_end   * 1000.0))

    nrows = k + (1 if showgeneQ else 0)
    fig, axs = plt.subplots(nrows, 1, figsize=(W, H * nrows), sharex=True)
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
        args = tuple(g) if is_list else (g,)

        plotf(*args,
              labels=at(mlabels, i, None),
              x_array=at(mx_array, i, None),
              dual_axis=dualQ,
              resolution=at(mresolution, i, 1),
              region_high=at(mregion_high, i, None),
              rname=at(mrname, i, 'Region'),
              invyQ=at(minvyQ, i, False),
              xlims=at(mxlims, i, (None, None)),
              ylims=at(mylims, i, (None, None)),
              xtitle=at(mxtitle, i, 'Index'),
              ytitle=at(mytitle, i, 'Value'),
              x_show=(i == k - 1),   # bottom plot only (among plotf panels)
              y_show=at(my_show, i, True),
              logyQ=at(mlogyQ, i, False),
              scale_matchQ=at(mscale_matchQ, i, False),
              saveQ=False,
              sname=at(msname, i, 'test'),
              ext=at(mext, i, 'pdf'),
              layout_rect=layout_rect,
              fig=fig,
              ax=ax,
              showQ=False)

    axs[-1].set_xlim(xk_start, xk_end)

    # If the gene axis used set_xticks([]), it would have cleared the shared ticks.
    # Restore auto ticks + ensure bottom axis is visible.
    if showgeneQ:
        axs[-1].xaxis.set_major_locator(mticker.AutoLocator())
        axs[-1].xaxis.set_minor_locator(mticker.NullLocator())
        axs[-1].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    fig.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1],
                        right=layout_rect[2], top=layout_rect[3], hspace=hspace)

    if at(msaveQ, 0, False):
        fig.savefig(f"figures/plot_{at(msname, 0, 'test')}.{at(mext, 0, 'pdf')}", bbox_inches="tight")

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
