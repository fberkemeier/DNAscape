# DNAscape NiceGUI interface

This folder contains the local DNAscape NiceGUI interface.

The interface includes the full navigation shell, responsive layouts, light
and dark themes, scientific controls, previews, tabs, expanders, and visible
action buttons. Simulation runs bounded constant or uploaded-profile jobs
through `dnascape.rsim`. Mapping resolves the selected target, route, and input forms,
automatically inspects uploaded profiles, and runs `dnascape.rmap`. Both live
workflows use background workers, progress reporting, and result plots in
Preview. Plot downloads support PNG, SVG, and PDF; selected raw outputs support
text, NPY, and NPZ downloads. Histogram settings keep the bin count and any
positive exact bin size synchronized, can draw touching or spaced bars, and let
the user choose whether an internal edge belongs to the left or right bin;
incomplete numeric edits leave the last valid preview in place. Application
Settings controls plot DPI and the compression format, with GZIP selected by default.
**Conversion memory limit (MB)** sets the estimated working-memory budget for
BigWig and bedGraph preparation, from 64 MB (default) to 4096 MB. The K562 ENCODE
example at 1 kb bins for all chromosomes needs about 322 MB; choosing 512 MB
allows that conversion. Related track array and download limits scale with the
budget. This setting does not reserve RAM, cap actual RAM use, or change upload
limits. Settings uses grouped switches and draft preferences; Apply saves changes.
Restore defaults fills the draft without clearing any data or results. Application restart and
Reset everything restore the 64 MB default.

You can switch pages while converting, mapping, simulating, or preparing a plot.
The sidebar shows a spinner beside that page while the job runs and a check when
it completes while another page is open. Selecting the page clears its check;
Stop immediately clears the indicator. Active jobs and unread results remain
available even when preserving page state is switched off.

Spatial profiles use the same curve renderer across all four workflow pages.
Mapping time and space-time outputs use heatmaps with position on X, time on Y,
and the mapped quantity on the colour scale. Axes stay at the plot edges when inverted.
Automatic X and heatmap Z bounds are shown in their editable fields, and Reset
restores the data-derived range. Plot titles are generated from the result.
Time and space-time firing-rate fits use Weibull without a shape selector.
Curve type supports Solid, Dashed, Dotted, and Dash-dot in Style and individual
Curves controls. Individual curves can inherit the shared setting or override it.
With exactly two combined curves, Plotting also offers **Two Y axes**, off by
default. Y1 is on the left and Y2 on the right, each with independent limits and
reset buttons; switching it off restores the shared Y scale.

Data Manager fills empty genomic bins with zero by default, including for the
ENCODE example. Zero and custom-value fills are applied after thresholding,
smoothing, and value scaling, so the requested fill is unchanged in the output.
Fill with minimum and Fill with maximum use each chromosome's finite values
after thresholding and smoothing and fill empty bins before value scaling.
These fills participate in the scaling calculation. If no finite values remain,
the conversion reports an error because the requested minimum/maximum is undefined.
Mapping preserves blank rows and leading bins and reports remaining missing
values by default. Interpolation and explicit fills retain bin positions;
dropping missing bins is unsupported in the interface and `rtrack`. Only trailing
padding of shorter columns in wide chromosome tables is removed. Existing files
with missing bins must be filled or regenerated before mapping.
The SNS-seq K562 BED example selects its declared enrichment-ratio column and
prepares Firing rate with 1 kb bins, maximum peak aggregation, zero background,
and no interpolation or smoothing. IGV display scores and coordinates are
excluded from the signal picker. Peak previews retain local maxima and background
at their original positions within the 500-point limit. Aggregation choices
distinguish scored BED peaks, unit-valued BED3 calls, and continuous bedGraph or
BigWig signals; peak-call gaps cannot be interpolated. Firing rate excludes centre
and Z-score scaling. SNS enrichment is relative origin activity and needs a
chosen calibration to express a firing rate. Repli-seq BigWig retains mean
aggregation and its existing continuous-signal operations and preview.
The OKseqHMM K562 OK-seq example links to the authors' hg19 fork-directionality
BigWig at 1 kb resolution, with read-count cutoff 10 and no 15 kb smoothing.
It selects Fork directionality, mean aggregation and Keep empty bins, preserving
uncovered bins as missing rather than treating them as balanced forks. The
source is available in the [OKseqHMM published K562 results](https://github.com/CL-CHEN-Lab/OK-Seq/tree/master/published_results/K562).
Mapping raw downloads start with TSV (the first format option) and compression
disabled, for both individual and batch outputs.
Simulation accepts up to 50,000 replicates. One-dimensional time courses and
memory-intensive position-by-time matrices have separate off-by-default
switches; S-phase duration, absolute completion time, origins per replicate,
inter-origin distances, and replication-timing standard deviation are standard
outputs. A memory-bounded
subset of at most 50 individual timing profiles is retained for optional
overlays.

Mapping aligns spatial inputs of different lengths automatically. Start and End
refer to the longest profile's chromosome coordinates; shorter profiles begin
at their first point and retain their bin width. At 1 kb/bin, 30,000–35,000 kb
uses that slice of the longest profile and the first 5,000 points of a shorter
profile, plotted at the selected chromosome positions. Mapping reports an error
if the requested width exceeds the shortest profile's length, or End lies beyond
the longest profile; it never silently shortens the selection. **Whole region**
uses the first points of every profile, from 0 to the shortest profile's length.
Shorter prepared profiles follow the same rule, with their position axes aligned
to the selected Start. Downloaded code uses the same validation and alignment.
Temporal inputs still need a consistent time grid.
For spatial fork speed from replication timing and constant firing, Start/End
crop the timing array before the fit; the constant applies throughout that region.
This route uses the earlier deterministic local speed iteration: increase speed
where predicted timing is late and decrease it where timing is early, retaining
the best evaluated profile. Its defaults are 10 updates, step 0.4, smoothing
over 100 bins, a 2000 kb neighbourhood, and initial speed 1.4 kb/min. Progress
and cancellation remain available; the route requires no compiled kernel or
time grid.
Spatial fork-speed fitting also offers **Maximum edge context (kb)**, default
1000 kb, matching spatial firing-rate fitting. On linear subregions it reflects
firing and speed beyond the endpoints, caps the context adaptively, and tapers
only the last few updates to reduce artificial endpoint spikes. Every position
remains refinable; timing MSE scores the core outside the inferred context.
Set the control to 0 for true chromosome ends. Periodic fits are unaffected,
and downloaded Mapping code retains the chosen value.
Spatial fork-speed fitting smooths its log-speed corrections and checks timing
MSE before accepting each update. If smoothing prevents improvement, it tries a
local correction with a reduced step as needed. This keeps profile-based firing
fits from repeatedly flattening the accumulated speed profile or increasing
MSE. All fitting routes retain their latest live MSE through intervening progress
messages, using the same scientific-number display as firing-rate mapping.
Live MSE also covers directionality-based fits and refinement stages with
console verbosity disabled.
Time-dependent speed uses the earlier compact temporal fit; space-time speed
uses the earlier alternating spatial and temporal fit, with controls for its
phase steps, iteration limits, smoothing, and update order. Periodic geometry
retains the constrained inverse because the earlier fitters support linear
geometry only. All Mapping runs show a persistent ETA and elapsed time:
"estimating" during preparation or quiet work, followed by an estimate when
solver progress or completed batch cases provide enough information.
Heatmap automatic axes end at the raster edges, without extra plot padding,
except for the next round boundary on zero-based position profiles (for example,
10,000 bins labelled 0 through 9,999 still show the 10,000 tick).
X and Y limits are editable for every plot, with Z (colour) limits for heatmaps.
Reset restores automatic ranges, including after an incomplete edit. Custom
aspect ratios in Style keep the preview width and adjust its height, with a
minimum height for the axes and colour scale. Reset restores the original size.
Hover coordinates use integers for whole-number axes, two decimals for fractional
axes, and scientific notation for tiny values. Simulation raw exports default to
uncompressed TSV, and origin efficiency uses an automatic upper Y limit with a
zero lower limit. Custom
heatmap axes crop or extend the visible domain without stretching the data;
genuinely missing cells stay transparent. Grid lines are drawn over the image.
Heatmaps automatically average large inputs into at most 300 time rows by 720
position columns, using bounded working tiles; original data and raw exports
keep their full resolution. SVG and PDF exports embed the heatmap as a PNG
while retaining vector axes, grid lines, colour scales, and labels.
Mapping options include **Show Plot**, enabled by default. Turning it off skips
preview generation while retaining mapping results and raw downloads; turning
it back on displays the completed result without rerunning the mapping.
Same-quantity reductions identify their source as space-time data so they cannot
be mistaken for a route that simply returns its target unchanged. Exact
same-quantity, same-form routes are excluded from the Mapping catalogue.

## Mapping code downloads

Enable **Dev Mode** in Settings to show the **Code** button beside the workflow
action. Mapping's menu exports the current controls as a Python script or a
runnable ZIP. The other Code buttons remain placeholders.

The ZIP contains `mapping.py`, the original input files (or an NPZ prepared
result), the current DNAscape package source, and pinned dependencies. Extract
it and run `python -m pip install -r requirements.txt`, then `python mapping.py`.
With uv installed, `uv run mapping.py` handles the script's declared dependencies.
Use Python 3.10 or newer. A script-only download uses the same relative input
paths and needs the package and inputs supplied by the ZIP.

Edit `MAPPING` for route, domain, input paths, selected columns/array keys,
constants, track processing, combinations, and optional setting overrides.
`RAW_EXPORT` includes the selected output, filename, format, compression switch,
compression format, and individual batch choices. `EXPORT_RAW` defaults to true.
Set `PLOT = True` for optional HTML plots using the app's ECharts specifications;
`PLOTTING` exposes the preview settings, colors, layout, and export DPI. The
bundle includes the installed ECharts renderer for offline plots, with PNG and
SVG downloads available in the HTML. A stochastic route with no fixed seed
produces a new realization on each run.

The shared Python workflows live in `dnascape.workflows`. The app's existing
service imports remain compatible and use the same implementations, validators,
scientific defaults, limits, and serializers. Code export snapshots controls
without running Mapping or replacing its displayed results.

## Install and launch

From the repository root:

The prototype requires Python 3.10 or newer (the scientific package itself
continues to support its existing Python range).

Python 3.13 is recommended for BigWig imports: the app installs `pybigtools`
on Python 3.10–3.13. Python 3.14 or newer can run the app with tables, NumPy
profiles, and bedGraph tracks; BigWig imports require a separately installed
compatible reader. The discontinued `gwseq-io-pp` package is no longer an
installation dependency.

```powershell
python -m pip install -r app/requirements.txt
python -m app
```

Open `http://127.0.0.1:8080` after the server starts. To choose another port:

```powershell
$env:DNASCAPE_APP_PORT = "8090"
python -m app
```

DNAscape does not launch a browser automatically. Set `DNASCAPE_APP_SHOW=true`
only when an automatic browser window is wanted.

Workflow pages load their modules when first opened. Starting the server does
not import every mapping, simulation, plotting, and conversion dependency.

The logos are served directly from `docs/source/_static/`; they are not copied
or duplicated in this folder.

## Application boundary

- `app/main.py` owns the NiceGUI process and persistent shell.
- `app/pages/` contains the seven workflow pages.
- `app/simulation_service.py` validates bounded scalar/profile jobs and bridges
  the Simulation page to `dnascape.rsim` without coupling scientific code to
  NiceGUI elements.
- `app/simulation_exports.py` lazily creates bounded plot and raw-data downloads
  from completed Simulation results.
- `app/mapping_catalog.py` presents the implemented `rmap` registry as target,
  route-family, input-form, and setting metadata for the interface.
- `app/mapping_uploads.py`, `app/mapping_service.py`, and
  `app/mapping_exports.py` provide bounded upload inspection, background
  Mapping execution, and rdata-compatible downloads.
- `app/data_conversion_service.py`, `app/bigwig_service.py`, and
  `app/remote_sources.py` implement the Data Manager pipeline. It separates
  automatic technical-format detection from the user-supplied biological
  meaning, reads BigWig and bedGraph tracks by genomic region, and accepts
  bounded public HTTP(S) links without allowing access to private network
  addresses. Genomic exports contain DNAscape-ready profile values rather than
  coordinate columns: a single-chromosome text file uses the chromosome as its
  header, while an all-chromosome text file is a labelled wide table and an
  all-chromosome NPZ stores one native-length array per chromosome key. Plain
  TSV is the default. Selected profiles can be smoothed, multiplied by a factor,
  rescaled into an explicit target interval (with optional inversion), or
  threshold-filtered without deleting or shifting samples; each value-scaling
  operation is mutually exclusive, and filtered values are represented as
  missing. Result plotting is an off-by-default, 500-point visual check.
- `app/session_cache.py` keeps a bounded, per-browser-tab inventory of
  uploads, prepared datasets, generated files, and completed workflows.
  Cached Mapping and Simulation results can be reopened without rerunning the
  scientific calculation. Oldest entries are removed automatically above 30
  items per category, 80 total items, or 256 MB retained memory.
- `app/preview_plotting.py` powers the shared result previews and standalone
  Plotting workflow, including exact-width and edge-aware histogram bins,
  editable X ranges, filled uncertainty bands, and a default positionable
  mean/median/standard-deviation summary.
- `app/static/styles.css` implements the DNAscape visual language and includes
  desktop, tablet, and mobile layouts.
