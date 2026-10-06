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
limits. Settings changes take effect immediately. Application restart and
Reset everything restore the 64 MB default.
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
Mapping raw downloads start with TSV (the first format option) and compression
disabled, for both individual and batch outputs.
Simulation accepts up to 50,000 replicates. One-dimensional time courses and
memory-intensive position-by-time matrices have separate off-by-default
switches; S-phase duration, absolute completion time, origins per replicate,
inter-origin distances, and replication-timing standard deviation are standard
outputs. A memory-bounded
subset of at most 50 individual timing profiles is retained for optional
overlays.

Mapping accepts spatial inputs of different lengths. The selected Start/End
window is capped to the shortest available spatial profile, keeping the declared
bin width. A longer chromosome upload supplies genomic coordinates; a shorter
plain profile is treated as local data beginning at Start. Prepared results with
position axes keep their explicit coordinates. No stretching or interpolation is
implied by this alignment. Temporal inputs still need a consistent time grid.
For spatial fork speed from replication timing and constant firing, Start/End
crop the timing array before the fit; the constant applies throughout that region.
This route uses the earlier deterministic local speed iteration: increase speed
where predicted timing is late and decrease it where timing is early, retaining
the best evaluated profile. Its defaults are 10 updates, step 0.4, smoothing
over 100 bins, a 2000 kb neighbourhood, and initial speed 1.4 kb/min. Progress
and cancellation remain available; the route requires no compiled kernel or
time grid.
Time-dependent speed uses the earlier compact temporal fit; space-time speed
uses the earlier alternating spatial and temporal fit, with controls for its
phase steps, iteration limits, smoothing, and update order. Periodic geometry
retains the constrained inverse because the earlier fitters support linear
geometry only. All Mapping runs show a persistent ETA and elapsed time:
"estimating" during preparation or quiet work, followed by an estimate when
solver progress or completed batch cases provide enough information.
Heatmap automatic axes end at the raster edges, without extra plot padding.
Requested heatmap bounds beyond the supplied data are limited to its extent;
inner crops are retained, and genuinely missing cells stay transparent.
Same-quantity reductions identify their source as space-time data so they cannot
be mistaken for a route that simply returns its target unchanged. Exact
same-quantity, same-form routes are excluded from the Mapping catalogue.

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
