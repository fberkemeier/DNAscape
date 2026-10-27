# DNAscape NiceGUI interface

This folder contains the local DNAscape NiceGUI interface.

The interface includes the full navigation shell, responsive layouts, light
and dark themes, scientific controls, previews, tabs, expanders, and visible
action buttons. Simulation runs bounded constant or uploaded-profile jobs
through `dnascape.rsim`. Mapping resolves the selected target, route, and input forms,
automatically inspects uploaded profiles, and runs `dnascape.rmap`. Both live
workflows use background workers, progress reporting, and result plots in
Preview. Plot downloads support PNG, SVG, and PDF; selected raw outputs support
TSV, TXT, CSV, and NPZ downloads. Histogram settings keep the bin count and any
positive exact bin size synchronized, can draw touching or spaced bars, and let
the user choose whether an internal edge belongs to the left or right bin;
incomplete numeric edits leave the last valid preview in place. Application
Settings controls plot DPI and the compression format, with GZIP selected by default.
In Cartesian plot previews, drag to pan within the default axis bounds. Hold left Ctrl
and scroll to zoom both axes, or hold left Ctrl and drag across a shaded region
to zoom only X (or draw an X/Y box for heatmaps). Axis fields and plot downloads follow the current view; the axis
reset arrows restore the default ranges. Zooming refines curves and heatmaps
from the underlying data, revealing more detail in the visible region.
Legends default to the top right. The fit-to-window and download buttons appear
at the bottom right inside the axes; axis lines stay visible when zoomed in.
**Circular** view offers **Track height (%)**, down to 1%, and **Inner radius (%)**
for the central opening. **Combine**, shown for multiple datasets, overlays their
tracks; otherwise they use separate rings. A mean, its standard-deviation band
and individual simulations share one track. Left Ctrl and scrolling magnifies the
view; Left Ctrl and dragging magnifies the smaller arc in its original angular
position, including across zero. Normal dragging pans around the circle. Hovering anywhere inside the
circle draws a dashed guide from its centre, one position readout beside the highlighted point, and
matching curve points. The floating box contains only coloured signal values.
The legend defaults to the top right; **Legend position** selects any corner and
stays fixed while zooming. Exported HTML uses the same controls.
Simulation distributions offer Histogram and Boxplot views.
Under Guides, **Hover** toggles value readouts, dashed guides, and matching
points on hovered curves. Floating boxes show only the datasets' Y values; for
heatmaps they show only Z, with X and Y readouts on the axes. Axis readouts use
the DNAscape theme colour. Axis-label crosses remove the corresponding titles.

Histograms combine the legend and **Statistics** in one panel, positioned at
the top right by default. With Statistics enabled and multiple combined
datasets, use its dropdown or click a bin to select a dataset. With Statistics
off, the legend lists every dataset and has no dataset selector. **Side by side** places
datasets within each shared bin interval, keeping bin size unchanged. Enabling
**KDE** replaces bars with density curves; the adjacent **Smoothness** multiplies
Scott's bandwidth (1 by default, larger values give smoother curves).
**Conversion memory limit (MB)** sets the estimated working-memory budget for
BigWig and bedGraph preparation, from 64 MB (default) to 4096 MB. The K562 ENCODE
example at 1 kb bins for all chromosomes needs about 322 MB; choosing 512 MB
allows that conversion. Related track array and download limits scale with the
budget. This setting does not reserve RAM, cap actual RAM use, or change upload
limits. Settings uses grouped switches and draft preferences; Apply saves changes.
**Field hover hints** under **Interface** enables brief field descriptions and is
on by default. Turn it off without affecting plot hover readouts or the separate
**Field help** option, which stays off by default.
Under **Visualisation**, **DNAscape theme** offers a colour picker and hex input for
the app accents, logo and browser-tab icon, with readable light and dark variants.
Restore defaults brings back the original DNAscape red.
This section also contains **Controls font size** and **Preview points**. The
point limit defaults to 1,000 per curve in the visible range and can be set from
100 to 10,000. Apply updates existing previews while keeping their zoom window.
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
Time and space-time initiation-rate fits use Weibull without a shape selector.
Curve type supports Solid, Dashed, Dotted, and Dash-dot in Style and individual
Curves controls. Individual curves can inherit the shared setting or override it.
In Analysis, combined layouts offer **Curves**, **Histograms**, or **Boxes**
controls for individual dataset names and colours. Histograms also expose
opacity and, with KDE on, thickness and type. Boxes expose outline thickness,
numeric X position, and an optional X label; each box is centered on its tick.
These per-dataset sections are hidden in separate layouts.
With exactly two combined curves, Analysis also offers **Two Y axes**, off by
default. Y1 is on the left and Y2 on the right, each with independent limits and
reset buttons; switching it off restores the shared Y scale.

The Analysis page combines plotting with independent moving-average or Gaussian
smoothing for each selected dataset. **Enable analysis** is off by default;
switching it on changes **Plot** to **Analyse** and reveals Pearson, Spearman,
and Kendall tau controls. **Show p-values** adds two-sided, unadjusted
p-values beneath the coefficients; self-correlations have no p-value. The
**Analysis report** below the plotting settings combines all selected methods
into one table with grouped correlation headers and dataset columns. The bottom
row has an editable report name, CSV/TSV selection, a download arrow, and a cross
to clear the preview while retaining uploaded data. The single file contains all
coefficients, paired sample counts, and optional p-values. CSV is the default.
Disabling analysis hides its correlation controls and report.
Pairs use matching array indices over the shared span
and exclude nonfinite values; inputs should have matching regions and resolutions.

**Data transforms** and **Region transforms**, below Analysis, follow Data
Manager's controls and work with or without correlations. Each section has a
**Dataset** menu for one dataset or all datasets. Data transforms provides
Python/NumPy formulas using **source**, optional **Combine** formulas using
**source1**, **source2**, etc., followed by smoothing, value scaling and inversion.
Region transforms selects a mask and changes its values; the default replaces
values below zero with zero. **Keep original profiles** retains the profiles
before the regional edit alongside the transformed ones.

**Analysis on mask only**, below the Region transforms switches, limits the
selected datasets' correlations and distribution statistics to their masked
samples. Pairwise comparisons use the shared finite masked samples, preserving
sample indices. Curve plots and raw downloads retain the full profiles.
**Export mask** adds a raw download in positions or intervals, using the same
integer-coordinate formats as Data Manager. Every run starts from the uploads;
the exported Python script reproduces both transform sections and mask-only
analysis.

Data Manager offers the same missing-value options for every input type:
Keep empty values, Fill with zero (default), Fill with a value, Fill with minimum,
Fill with maximum, and Report error. Full tables apply the policy to numeric
signal columns while preserving labels and coordinates. Zero and custom-value
fills are applied after thresholding, smoothing, and value scaling, so the
requested fill is unchanged in the output.
Below Configure, **Region transforms** is collapsed and off by default. It runs
after Configure and **Data transforms**. A **mask** defines where changes apply;
the **transform** defines what happens to those values. Enable **Apply transform**
to edit the selected regions. The default replaces values below zero with zero.
Under **Select regions / define mask**, select values above/below a threshold,
within or outside two thresholds, at a distance from the profile's minimum or
maximum, missing values, **Position ranges**, **Upload profile**, or the whole region.
**Position ranges** accepts comma-separated intervals such as `(1000,2000), (4000,5000)`.
Starts are included and ends excluded. Its text field grows as text wraps;
**Coordinates** defaults to bp and also accepts kb. Empty ranges apply no mask.
**Upload profile** detects one position column or two interval columns;
**Coordinates** defaults to bp and also accepts kb. With no mask uploaded,
that selection leaves the data unchanged. **Invert mask** selects the complement
within the configured Start/End region.
The options remain visible with **Apply transform** off. In **Plot settings → Guides**,
**Mask** shades the selected regions in light grey, preserving
the pre-edit selection and native-bin detail on zoom. Enable **Legend** to name
the plotted signal and mask separately. Display settings do not change the data.
**Export mask**, beside **Apply transform** in Controls, adds a headerless raw-data
download of selected positions or genomic intervals as whole numbers. Preview
sampling never reduces the exported positions.
You can select **Export mask** in advance; it produces no mask output while
**Apply transform** is off.

Region transforms replace with a value or NaN, average each contiguous region,
interpolate from surrounding values, add zero-mean Gaussian noise, match a
minimum/maximum, multiply, or add. Reference extrema use the converted values
before the region transform, either outside the mask or across the whole
profile. Edge interpolation uses the nearest available endpoint; selecting the
whole profile requires a non-interpolating operation. Noise accepts a standard
deviation and optional reproducible seed. Each chromosome and time slice is
handled independently; coordinates and bin counts are preserved. NaNs created
by a transform are retained in previews and downloads. Genomic intervals are
half-open (start included, end excluded) and select overlapping bins. Mask
coordinates retain the source start and resolution. Chromosome-labelled masks
stay within their chromosomes.

Choose **Modulate with profile** to combine the converted signal with a separate
prepared signal or annotation profile, such as gene locations, RNA-seq, ATAC-seq
or ChIP-seq. Choose **Whole region** to
affect every bin, or retain any existing region selection. The modulation profile
must match the converted bin count and matrix dimensions; row/column vectors
are equivalent, but matrices are not broadcast or resampled. Explicit
coordinates are checked when both profiles supply them. Automatic selection
matches chromosome columns or NPZ arrays by name; wide chromosome tables retain
leading/internal missing bins and trim only trailing padding.

**Combine using → Formula** offers weighted sums, multiplication and division.
Each input has its own signed multiplier, exponent and function (natural,
base-2 or base-10 logarithm, or exponential), with a live formula preview.
Minimum, maximum and convolution remain separate methods. **Profile scaling** can preserve original
values or normalize the modulation profile to mean 1, maximum magnitude 1,
or a 0–1 interval before combining. A zero weight omits a term from a weighted
sum. Logarithms require positive selected values. Division by zero reports an error by default, with
options to keep the signal or set NaN. Missing modulation values can report an
error, use zero, or remain NaN.

Convolution uses the uploaded profile as a centred kernel along each ordered
profile (spatially for space-time matrices), retaining the original size. Kernel
normalization is on and boundaries reflect by default; periodic, nearest-value
and zero-padding boundaries are also available. Convolution requires uniformly
spaced bins and finite inputs, or explicit zero-filling of missing values.
Only selected output bins change. Previews, raw exports and format changes use
the same result. Provenance includes the operation and a hash of the uploaded
modulation profile; auxiliary uploads are not duplicated into Cached Files.

Fill with minimum and Fill with maximum use each profile or chromosome's finite values
after thresholding and smoothing and fill empty bins before value scaling.
These fills participate in the scaling calculation. If no finite values remain,
the conversion reports an error because the requested minimum/maximum is undefined.
Report error rejects missing samples, including empty bins before interpolation
and thresholded samples. GEO 16-fraction Repli-seq text matrices (`.mat`/`.mat.gz`)
can be exported as tables or converted into chromosome-specific timing profiles
using a cumulative sigmoid T50 fit or signal-weighted mean. Missing-value handling
applies after timing extraction. See [Managing data](../README.md#managing-data)
for a K562 BigWig conversion walkthrough.
The ENCODE K562 example uses UCSC's current `hgdownload.gi.ucsc.edu` download
host. Pasted HTTPS links using the older `hgdownload.soe.ucsc.edu` or
`hgdownload.cse.ucsc.edu` aliases use the current host with the same file path
and query. HTTPS certificate verification remains enabled. Connection timeouts
and certificate failures are reported separately.
Mapping preserves blank rows and leading bins and reports remaining missing
values by default. Interpolation and explicit fills retain bin positions;
dropping missing bins is unsupported in the interface and `rtrack`. Only trailing
padding of shorter columns in wide chromosome tables is removed. Existing files
with missing bins must be filled or regenerated before mapping.
The SNS-seq K562 BED example selects its declared enrichment-ratio column and
prepares Initiation rate with 1 kb bins, maximum peak aggregation, zero background,
and no interpolation or smoothing. IGV display scores and coordinates are
excluded from the signal picker. Peak previews retain local maxima and background
at their original positions within the configured preview point limit. Aggregation choices
distinguish scored BED peaks, unit-valued BED3 calls, and continuous bedGraph or
BigWig signals; peak-call gaps cannot be interpolated. Initiation rate excludes centre
and Z-score scaling. SNS enrichment is relative origin activity and needs a
chosen calibration to express an initiation rate. Repli-seq BigWig retains mean
aggregation and its existing continuous-signal operations and preview.
The OKseqHMM K562 OK-seq example links to the authors' hg19 fork-directionality
BigWig at 1 kb resolution, with read-count cutoff 10 and no 15 kb smoothing.
It selects Fork directionality, mean aggregation and Fill with zero. Choose
Keep empty values to retain uncovered bins as missing. The
source is available in the [OKseqHMM published K562 results](https://github.com/CL-CHEN-Lab/OK-Seq/tree/master/published_results/K562).
Mapping raw downloads start with TSV (the first format option) and compression
disabled, for both individual and batch outputs.

Raw-file rows include an **Open in…** action between Download and Remove.
Choose Data Manager, Mapping, Simulation, or Analysis to add the file to that
page's uploads, including the current page. The selected name, file type, and
compression are retained. In Mapping, choose **Target**, **Route**, and the
specific **Input**; **Dependence** appears when the target supports several
choices. The page adopts these choices only when an input is selected.
Simulation selects a matching input when known, or asks which input to use.
Mask-position and mask-interval rows open directly in **Region transforms** with
**Upload profile** selected and bp coordinates. Simulation asks whether the mask
belongs to initiation rate or fork speed. Importing a mask leaves **Apply transform**
unchanged. Existing data uploads remain, and repeated names receive a numbered
suffix. Individual-profile ZIPs open as separate files and remain excluded from
Cached Files.
In Data Manager, the selector beside each generated file chooses its plot;
the latest file is selected after conversion, and plot exports use that selection.
Expand Plot settings below the preview to adjust its appearance and axes or
switch to a histogram or KDE curve. These settings also apply to plot downloads.

Simulation accepts up to 50,000 replicates. Stochastic fork movement is
optional. Stochastic speeds use positive lognormal draws with the local input
mean and the requested standard deviation; temporal
speeds are integrated through time-slice boundaries while crossing bins. This
avoids the near-zero speed artifacts of positive Gaussian draws. Turning
stochastic movement off preserves the existing models and seeded results.
In **Fork stalling**, **Shared between directions** uses the same genomic stall
positions and recovery delays for both directions. Off by default, the two
directions stall independently; this does not couple distant sister forks.

One-dimensional time courses and memory-intensive position-by-time matrices have
separate off-by-default
switches; S-phase duration, absolute completion time, origins per replicate,
inter-origin distances, and replication-timing standard deviation are standard
outputs. A memory-bounded
subset of individual timing profiles is retained up to **Profiles shown**
(10 by default, at most 50, subject to the memory limit). Reducing this count
releases the omitted profiles; run again after increasing it to retain more.
Enabling **Individual profiles** or **Mean ± standard deviation** adds the
corresponding downloads below the main raw data. Individual profiles download
as a ZIP containing one file per displayed replicate (TSV by default, with
TXT, CSV and NPZ also available). Compression stays on for this archive, whose
name includes the actual replicate count. Individual profiles and their ZIPs
are excluded from Cached Files; cached simulation results keep aggregate
statistics only.
**Track time courses** also records **Origin firing (time)** (mean actual
initiations per sampling interval) and **Replicon length** (a pooled histogram
of the completed DNA domains copied from individual fired origins, in kb).
Periodic domains join across the region boundary; unfinished replicons are
excluded. Replicon samples have a bounded retention limit with an explicit
truncation warning. All time-only output names carry **(time)**.
**Track space-time outputs** adds six matrices, including **Total fork density
(space-time)** as the sum of the left- and right-moving densities. Initiation
events count fired origins, excluding passively replicated firing proposals.

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
1000 kb, matching spatial initiation-rate fitting. On linear subregions it reflects
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
messages, using the same scientific-number display as initiation-rate mapping.
Live MSE also covers directionality-based fits and refinement stages with
console verbosity disabled.
Time-dependent speed shares the guarded temporal stage of sequential space-time
fitting, holding the spatial component constant. Cached native initiation
integrals and analytic derivatives refine linear or curved temporal profiles
without repeated finite-difference predictions. Space-time speed uses the
alternating spatial and temporal fit, with controls for its
phase steps, iteration limits, smoothing, and update order. Periodic geometry
retains the constrained inverse because the earlier fitters support linear
geometry only. All Mapping runs show a persistent ETA and elapsed time:
"Initial prediction" appears on every route. Estimates update during the first
prediction as measured work becomes available, then refine from fit iterations.
Preparation remains "estimating" until there is enough measured work. Status,
MSE, ETA and elapsed time share the same compact line across all routes.
Automatic X axes end at the first and last data coordinates, including heatmap
raster edges. Tick labels use clean intervals independently of those limits;
an endpoint receives a label only when it falls on that interval.
X and Y limits are editable for every plot, with Z (colour) limits for heatmaps.
Reset restores automatic ranges, including after an incomplete edit. Custom
aspect ratios in Style range from 0.2 to 8 and adjust the height to the preview
width without a minimum-height clamp. The default and reset aspect ratio is 2.5.
Line previews draw up to 1,000 native points from the visible window by default,
with the budget set in Visualisation settings. Detail is prefetched for one
visible window on either side, with a bounded coarse overview elsewhere.
Small pans reuse that detail; deeper zooms load more native samples. Browser
gestures are drawn once per frame, and delayed server responses preserve the
latest viewport. Native-data reads run in one background worker that coalesces
requests, keeping the interface responsive. Plain single-column text outputs
use a compact row index to avoid rescanning the entire file for each view.
Heatmaps retain their full-region raster around the detailed tile;
missing fine cells stay transparent.
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

## Developer code downloads

Enable **Developer mode** under Interface in Settings, then click **Code** on
Data Manager, Mapping, Simulation or Analysis (Plotting). Each button downloads
one editable `.py` file directly. Exporting code reads the current controls and
does not start the page's conversion, fit, simulation or analysis.

Use Python 3.10 or newer and install the matching DNAscape checkout, for example
`python -m pip install /path/to/DNAscape`. The scripts need the DNAscape package,
not the app or NiceGUI. Put the original source files listed at the top of the
script in `inputs/` beside it, or edit `INPUT_DIRECTORY` to point to their folder.
The original filenames and selected columns/arrays are retained; colliding
names use numbered source subfolders. Prepared Mapping arrays are included in
the script because they have no original source file. `OUTPUT_DIRECTORY`
defaults to `outputs/` beside the script, independently of the working directory.

- `data_manager.py` reproduces each source's configuration, binning, value
  processing, individual or combined transform formula, region transforms and raw
  export preferences, including exported mask positions or intervals.
- `mapping.py` exposes `MAPPING` and `RAW_EXPORT`: route, input forms and
  selections, region/resolution, fitting settings and overrides, target bounds,
  initial guesses, region transforms, batch combinations, output selection and export
  preferences. Its input preparation uses the same workflow as the page.
- `simulation.py` retains scalar/profile inputs, batch combinations, region and
  resolution, periodicity, random seed, stochastic movement, stalls, firing
  factors, tracking and separate initiation-rate/fork-speed masks and transforms. Selected
  raw data, enabled standard deviation/individual profiles and mask exports
  use the same serializers as the page. A seed of `None` produces a fresh run.
- `plotting.py` reproduces selected source profiles, smoothing, transformations,
  correlation methods and report options, combined or separate plot settings,
  curve/distribution styles, axes, guides, fill and circular views. It exports
  processed data and standalone HTML plots with PNG/SVG download controls.
  The HTML loads the pinned ECharts renderer when opened in a browser.

Data Manager, Mapping and Simulation scripts generate data without rendering
plots. Enabled histogram statistics are exported as data. Plot generation is
included only in the Plotting script. Edit `RECIPE` (or Mapping's `MAPPING` and
`RAW_EXPORT`) to change the operation, inputs or exports, then run
`python data_manager.py`, `python mapping.py`, `python simulation.py` or
`python plotting.py`.

The shared, headless implementations live in `dnascape.workflows`; the app and
the downloads use the same validation, scientific procedures and serializers.
For compatibility, Python settings and recipes retain identifiers such as
`DataFilterSettings`, `data_filter` and `filter_positions`. These represent the
region transforms and their masks; existing scripts and exported filenames keep
their established names.

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

Startup warms the Mapping and Data Manager modules; other workflow modules and
all page controls load when first opened. DNAscape keeps its own Python bytecode
in the per-user local cache, avoiding repeated reads from synced source folders
such as Dropbox. Source changes still invalidate the cache automatically.
The first launch populates it; subsequent launches reuse it. An explicit
`PYTHONPYCACHEPREFIX` takes precedence, and `DNASCAPE_BYTECODE_CACHE=off` restores
Python's default cache location.
Styles, scripts and logos also use local copies that refresh when the shipped
files change, avoiding slow synced-file reads on the first page request.

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
  rescaled into an explicit target interval, independently inverted, or
  threshold-filtered without deleting or shifting samples; each value-scaling
  operation is mutually exclusive, and filtered values are represented as
  missing. Result plotting is an off-by-default, 500-point visual check.
- `app/session_cache.py` keeps a bounded, per-browser-tab inventory of
  uploads, prepared datasets, generated files, and completed workflows.
  Cached Mapping and Simulation results can be reopened without rerunning the
  scientific calculation. Oldest entries are removed automatically above 30
  items per category, 80 total items, or 256 MB retained memory.
- `app/preview_plotting.py` powers the shared result previews and standalone
  Analysis workflow, including exact-width and edge-aware histogram bins,
  editable X ranges, filled uncertainty bands, and a default positionable
  mean/median/standard-deviation summary.
- `app/static/styles.css` implements the DNAscape visual language and includes
  desktop, tablet, and mobile layouts.
