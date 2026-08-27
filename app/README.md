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
Settings controls plot DPI and raw compression, with GZIP selected by default.
Simulation accepts up to 50,000 replicates. One-dimensional time courses and
memory-intensive position-by-time matrices have separate off-by-default
switches; S-phase duration, absolute completion time, origins per replicate,
inter-origin distances, and replication-timing standard deviation are standard
outputs. A memory-bounded
subset of at most 50 individual timing profiles is retained for optional
overlays.

## Install and launch

From the repository root:

The prototype requires Python 3.10 or newer (the scientific package itself
continues to support its existing Python range).

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
