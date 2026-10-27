NiceGUI App
===========

Run the local interface from the repository root:

.. code-block:: bash

   python -m pip install -r app/requirements.txt
   python -m app

Open ``http://127.0.0.1:8080`` after the server starts. DNAscape does not
launch a browser automatically; set ``DNASCAPE_APP_SHOW=true`` to opt in.
Use the app for inspection and short interactive runs; use Python scripts or
notebooks for reproducible analyses, sweeps, and publication pipelines.

Switching pages leaves conversions, mapping, simulations, and plot preparation
running. A sidebar spinner marks each active job and becomes a check if it finishes
while another page is open. Selecting that page clears the check; Stop clears the
indicator immediately.
Active jobs and unread results are retained even when preserving page state is off.

Pages
-----

Overview
   Navigate between Data Manager, Mapping, Simulation, Analysis, Cached Files,
   and Resources.

Data Manager
   Add a local file or a bounded public HTTP(S) link. The app automatically
   detects CSV, TSV, TXT, NPY, NPV, NPZ, bedGraph, and BigWig formats, while the
   biological role is selected with **Prepare as**. For genomic tracks, choose
   one chromosome and an optional half-open interval, or prepare every
   chromosome, then configure binning, aggregation, missing bins, interpolation,
   centred smoothing, threshold masking, and one mutually exclusive value-scaling
   operation: target interval, multiplier, centre,
   Z-score, or maximum absolute. A target interval is defined explicitly by its
   target minimum and target maximum. **Invert** reverses high and low signal
   values independently of the selected scaling. Thresholds are evaluated before smoothing
   and value scaling.
   Reusable TSV, TXT, CSV and NPZ spatial profiles also expose **Region & binning**.
   Choose a chromosome header or NPZ array with **Profile**, or use **Chromosome**
   for tables containing several chromosomes in rows. Supplied coordinates are
   retained. Otherwise, **Source start (bp)** and **Source resolution (bp)** define
   the original grid, initially 0 bp and 1,000 bp. Start, End, output bin size and
   aggregation apply before value processing, data transforms and region transforms.
   **Apply configuration** bypasses these steps when off. With multiple sources,
   untouched region limits use each source's full extent.
   The ENCODE K562 Repli-seq example uses ``hgdownload.gi.ucsc.edu``.
   Older HTTPS links to ``hgdownload.soe.ucsc.edu`` or
   ``hgdownload.cse.ucsc.edu`` use this current host with their file paths and
   queries preserved. HTTPS verification stays enabled; connection timeouts
   and certificate errors are reported separately.
   **Decimal places** sits above **Missing values**, with optional bounds below it.
   **Report error** stops conversion and reports missing samples. The reset
   button in **End** restores Start and End to the selected chromosome's
   available extent. Changes that allow or produce values outside the selected
   biological domain switch **Prepare as** to **Generic profile**, preserving
   the edited settings. Smoothing, binning and display changes retain the type.
   GEO Repli-seq text matrices from GSE137764 (``.mat`` or ``.mat.gz``) are
   transposed automatically into genomic-bin rows and S1-S16 columns. The
   default **Full table** selection retains chromosome, start, and end in TSV
   or CSV exports. Choose **All** or an individual S-phase column for numeric
   profiles. These raw exports retain the supplied fraction signals.
   **Replication timing (S1–S16)** instead extracts a chromosome-specific
   profile at the source's native resolution. **Sigmoid fit (T50)** fits the
   cumulative normalized fraction signal and returns the 50% replication time,
   following `Zhao, Sasaki and Gilbert (2020)
   <https://doi.org/10.1186/s13059-020-01983-8>`_. The sixteen intervals are equally
   spaced over an editable S-phase duration, defaulting to the paper's assumed
   600 minutes. **Signal-weighted mean** is also available, using the fraction
   interval midpoints. Incomplete, zero-signal, and failed-fit bins, and genomic
   gaps, retain their original positions. The chosen missing-value policy applies
   to these timings after extraction, defaulting to Fill with zero. Exports use minutes,
   chromosome-labelled headers, and chromosome-specific filenames. Mapping
   resolution should match the recorded source bin width (50 kb for H1).
   The **GEO 16-fraction Repli-seq (MAT) - H1** example fills the public GEO link
   and selects timing extraction after inspection.
   Binary MATLAB files are not included in this format support.

   All input types, including MAT timing profiles and full tables, offer the same
   missing-value options: Keep empty values, Fill with zero (the default), Fill
   with a value, Fill with minimum, Fill with maximum, and Report error.
   Full tables apply these policies to numeric signal columns, preserving labels
   and genomic coordinates. Report error rejects missing samples, including
   empty track bins before interpolation and values removed by thresholding.
   Thresholded samples remain on the original coordinate grid. Zero and custom
   fills are applied after smoothing and value scaling; Keep empty values preserves
   missing values. Fill with minimum and Fill with maximum instead use each
   chromosome's finite values after thresholding and smoothing, fill empty bins
   before scaling, and participate in the scaling calculation. They report an
   error if no finite values remain. Mapping reports remaining missing values by
   default and
   never drops leading or internal bins. Dropping missing bins is unsupported.
   The same polishing operations are available for selected space, time, and
   space-time profiles. Generated files contain regularly spaced DNAscape profile
   values, not source
   coordinate columns. A one-chromosome text export uses that chromosome as its
   header; an all-chromosome text export is a wide table with chromosome
   headers, while NPZ keeps one native-length array under each chromosome key.
   Formats that cannot retain chromosome identity are not offered for an
   all-chromosome export. TSV is the default. **Show plot** is enabled by default;
   zooming reveals more detail in the converted data. Generated files remain
   available in Cached Files for the browser tab.

   **Region transforms**, below Configure, starts collapsed with **Apply transform** off.
   It edits values after Configure and **Data transforms**. A mask defines where
   changes apply, and the transform defines what happens there. Its default selection is
   values below zero, replaced with zero. Other selections include values
   above/below a threshold, within or outside an inclusive interval, a threshold at a distance
   above the converted minimum or below its maximum, NaN regions, an uploaded
   genomic positions/intervals, and the whole region. An absent mask upload
   leaves the data unchanged. Upload one column of positions or two columns of
   half-open interval starts and ends; the layout is detected automatically.
   **Coordinates**, beside the upload, chooses bp (default) or kb. Intervals
   select overlapping source regions; optional chromosome labels restrict the
   selection. **Export mask**, beside **Apply transform** in Controls, adds one export row beneath the
   selected raw-data file, with a position-list or genomic-interval choice
   beside its file type.
   **Export mask** can be selected in advance while **Apply transform** is off;
   mask output is produced only when the transform is applied.
   A mask row's **Open in** action sends it directly to **Region transforms**
   on the chosen page, selects **Upload profile** and uses bp coordinates.
   Simulation asks whether it is an initiation-rate or fork-speed mask.
   Other raw files opened in Mapping offer **Target**, **Route** and **Input**,
   with **Dependence** where relevant. These choices are applied when an input
   is selected; cancelling preserves the current Mapping configuration.

   **Select regions / define mask** groups the selection controls. Choose
   **Upload profile** to upload coordinates, or **Position ranges** to enter
   intervals such as ``(1000,2000), (4000,5000)`` directly. The interval field
   grows as text wraps and has its own **Coordinates** selector, defaulting to bp.
   Starts are included, ends excluded, and an empty field applies no mask.
   **Invert mask** selects
   the complement within the configured Start/End region.

   Region transform controls remain editable while **Apply transform** is off. In
   **Plot settings → Guides**, enable **Mask** to shade the applied selection
   in grey. With **Legend** enabled, name both the signal and mask. Plot
   previews refresh up to 1,000 native curve points within the visible window by default
   after zooming or panning. Detail is prefetched beyond the visible area, and
   small pans reuse the buffered samples. A coarse full-region overview remains
   available while finer curve or heatmap detail loads in the background.
   Gestures draw once per animation frame; delayed updates cannot reset a newer
   viewport. Completed drags cannot trigger an accidental navigation click.
   Aspect ratios from 0.2 to 8 set the preview's
   width-to-height ratio directly.

   Operations include replacing with a value or NaN, the mean of each contiguous
   region, interpolation from surrounding endpoints (linear, smooth PCHIP,
   nearest, previous, or next), Gaussian noise, matching a minimum or maximum,
   multiplication, and addition. Reference extrema come from the converted
   profile before the region transform, including or excluding selected regions.
   Gaussian noise accepts a standard deviation and optional random seed.
   Interpolation at an edge uses the nearest available endpoint and requires
   at least one unselected finite value. Chromosomes and space-time columns
   are processed independently, retaining their grid. Previews and every raw
   export format use the transformed result, including any new NaNs.

   **Modulate with profile** accepts gene annotations or signals such as RNA-seq,
   ATAC-seq and ChIP-seq. Its editable Python/NumPy formula defaults to
   ``source * profile``: **source** is the configured and transformed signal,
   and **profile** is the uploaded modulation array. Arrays align from their
   beginning without resampling or broadcasting. Different sizes, coordinates
   or chromosomes produce a yellow warning; only the overlapping samples are
   combined and unmatched source values remain unchanged. Formula reductions
   use that overlap, while only positions within the mask are replaced.
   Errors preserve the previous generated result. Previews and raw downloads
   use the same modulated result.

   A curated UCSC ENCODE DCC K562 Repli-seq BigWig link is available as a test
   source. Its wavelet signal is a relative timing score (larger values are
   earlier), not an absolute replication time in minutes; the Data Manager does
   not silently change that scientific meaning.

   The K562 SNS-seq BED example selects column 4 (average enrichment ratio),
   excluding IGV display scores and coordinates. Prepare as Initiation rate uses
   1 kb bins, maximum aggregation to retain peak heights, zero background,
   no interpolation, and no smoothing. Its enrichment values represent relative
   origin activity; an initiation-rate calibration remains an explicit scaling choice.
   Scored BED peaks offer maximum, mean, integrated signal, and coverage;
   BED3 calls offer presence, covered bases, and coverage. Continuous bedGraph
   and BigWig retain their supported signal aggregations. Interpolation is
   unavailable between called peaks, and initiation-rate scaling excludes centre
   and Z-score. BED peak previews retain local extrema at their original
   coordinates within the configured preview point limit. Repli-seq BigWig conversion and
   preview behavior are unchanged.

   The OKseqHMM K562 OK-seq BigWig example provides hg19 fork directionality
   at 1 kb resolution, with read-count cutoff 10 and no 15 kb smoothing. It
   selects Fork directionality, mean aggregation and Keep empty bins, retaining
   uncovered bins as missing. The track is hosted in the authors'
   `published K562 results <https://github.com/CL-CHEN-Lab/OK-Seq/tree/master/published_results/K562>`_.

Mapping
   Configure ``rmap`` inputs, regions, routes, and preview settings, then run
   the resolved route in a cancellable background worker.

   Mapping settings put fit iterations before step size and start simulation
   ensembles at 100 runs. **Advanced** pairs **MSE threshold** with **Missing values**,
   neighbourhood range with maximum edge context, and **Initiation rate floor** with
   deviation threshold, followed by a full-width **Show plot** row. Route-specific
   controls appear only when needed. The initiation rate floor accepts the rate
   directly (for example, ``1e-15``), preserving the solver's numerical limit. Optional
   numerical fields use automatic values when blank; spatial overrides follow
   the main fit controls until edited. A zero MSE or deviation threshold disables
   that threshold. **Stall recovery** enables the recovery-time control.

   **Show plot** under Advanced is enabled by default. Turn it off to skip
   rendering the preview while still computing and retaining all results and
   raw downloads. Turning it back on displays the result without rerunning.

   Fitting routes keep their latest MSE visible between progress updates,
   including directionality fits and refinement with console verbosity disabled. Spatial
   fork-speed fitting smooths corrections and accepts only updates that improve
   timing MSE, including when firing is supplied as a spatial profile.

   Spatial fork-speed fitting offers **Maximum edge context (kb)** with the same
   1000 kb default as spatial initiation-rate fitting. Linear subregions use reflected
   firing and speed with an adaptive context cap and a narrow endpoint update
   taper to reduce artificial speed spikes. Every bin remains refinable; timing
   MSE scores the core outside that context. Set the value to 0 for true chromosome
   ends. Periodic fits are unaffected, and downloaded code keeps the chosen value.

   Periodic spatial initiation-rate fits also regularize log-rate curvature around
   the ring, including the join. This is applied to the seed and each fitting
   update before predicting timing; observed timing is unchanged. The smoothing
   scale is 5% of the typical fork-travel distance, bounded between one bin and
   1/32 of the region (one bin for very short regions), and is reported as
   ``fit_periodic_smoothing_kb``. The initiation-rate preview closes the smooth model
   curve at the period end, while downloads retain only the original bins.
   First and last stored bins are neighbouring positions, not duplicate endpoints.
   In Python, ``map_timing_firing(..., stabilize=False)`` retains the previous
   unconstrained periodic updates.

   Periodic mapping scores every valid spatial bin equally, without endpoint
   cropping, reflected padding or tapered updates. Spatial smoothing and
   regularization use the same weight at the join as elsewhere on the ring.
   Constrained fork-speed fits use equally spaced circular spatial knots;
   their temporal interpolation remains non-periodic.

   Profiles of different lengths align automatically. Start/End and plot positions
   refer to the longest profile's chromosome coordinates: at 1 kb/bin,
   30,000–35,000 kb selects that chromosome window and the first 5,000 points of a
   shorter profile. The requested width must fit the shortest profile, and End
   must lie within the longest profile; invalid selections produce an error.
   **Whole region** uses the first points of every profile, from 0 to the shortest
   profile's length. Prepared profiles retain their bin width and shorter position
   axes align to Start. Downloaded Mapping code follows the same rules.

   Enable **Developer mode** in Settings and use **Code** to download a single
   ``mapping.py`` file. Edit ``MAPPING`` for inputs and scientific settings and
   ``RAW_EXPORT`` for output selection, filenames, formats, compression and batch
   choices. Exporting code does not start Mapping or replace its results.

Developer code downloads
   **Code** is also available on Data Manager, Simulation and Analysis
   (Plotting), producing ``data_manager.py``, ``simulation.py`` and
   ``plotting.py``. Each script contains the current source selections and
   applicable processing/export settings. Data Manager includes per-source
   configuration, transform formulas and region transforms; Simulation includes batch
   inputs, movement, stalls, firing factors, tracking and separate input masks and transforms;
   Plotting includes smoothing, transformations, correlation reports and all
   combined/separate chart settings.

   Install the matching DNAscape checkout with Python 3.10 or newer. The scripts
   use ``dnascape.workflows`` and do not import the app or NiceGUI. Put the
   original files listed in the script in ``inputs/`` beside it, or edit
   ``INPUT_DIRECTORY``. Filenames and column/array selections are preserved;
   prepared Mapping data are included directly. ``OUTPUT_DIRECTORY`` defaults
   to ``outputs/`` beside the script, regardless of the working directory.

   Only Plotting includes plot generation, writing standalone HTML with PNG/SVG
   download controls; the HTML loads a pinned ECharts renderer when opened.
   The other scripts export their data and enabled supplemental files without
   rendering charts. A stochastic recipe with seed ``None`` produces a fresh
   realization. Downloading code never starts the corresponding page workflow.

Simulation
   Run bounded constant or uploaded initiation-rate and fork-speed fields through ``rsim``.
   Under **Origin & fork behaviour**, **Firing factor** places Fork coupling next
   to Finite pool, with Pool size and Recycle rate below. **Fork movement** offers
   separate Fork stalling and Stochastic movement switches, both off by default.
   Fork stalling reveals the physical stall rate, **Shared between directions**,
   and Stall recovery option. Shared stalls use identical genomic positions
   and recovery delays for both directions; off by default, directions stall
   independently. Stall recovery is off by default. Enable it to set a recovery time;
   with recovery off, stalls are permanent. Disabling stalling resets its
   rate to zero. Stochastic movement reveals Standard deviation (kb/min), applied
   around the local fork speed for constant, spatial, temporal, and space-time
   inputs. Positive lognormal speeds preserve the requested mean and standard
   deviation; temporal speeds are integrated during bin crossings. With
   stochastic movement off, existing models and seeded results are unchanged.
   **Fix seed** makes the simulation reproducible, including these movement choices.
   Progress appears in Preview while the scientific worker runs. Completion
   times, origins per replicate, and pooled inter-origin distances are standard
   ensemble outputs and do not require temporal tracking. Enable **Track time
   courses** for one-dimensional time summaries, including **Origin firing
   (time)** (mean actual initiations per sampling interval), and the **Replicon
   length** distribution. Replicon lengths measure the completed DNA domains
   copied from individual fired origins, joining domains across periodic
   boundaries and excluding unfinished domains. Samples have a bounded
   retention limit, with a warning if truncated. Time-only output names carry
   **(time)**. Enable **Track space-time outputs** for the six position-by-time
   matrices, including **Total fork density (space-time)**, the sum of the two
   directional densities. Initiation-event heatmaps include only origins that
   actually fired. The app validates
   actual retained-array sizes before allocation. Inter-origin distances use
   ``rsim`` values in kb, appear as a histogram by default, and can be switched
   to a box plot without rerunning the simulation. Very large IOD pools retain a
   deterministic bounded prefix and report that truncation explicitly.

   Every run calculates the population standard deviation of replication timing
   across the full ensemble. In the default replication-timing line view,
   **Plot settings** can show it as a filled mean ± standard-deviation band.
   **Profiles shown** limits how many individual timing profiles are retained
   (10 by default, at most 50, subject to the memory limit). These controls can
   be configured before simulating. Reducing the count releases omitted
   profiles; increasing it requires another run to retain more. The standard
   deviation always describes the full ensemble.

   Enabling either overlay adds its download below the main raw data. Standard
   deviation downloads as a separate profile. Individual profiles download as
   one ZIP, containing a TSV per displayed replicate by default; TXT, CSV and
   NPZ are also available. Compression is locked on for this archive, and its
   filename includes the actual replicate count. Individual profiles and their
   archives never enter Cached Files; saved workflow results retain aggregate
   statistics only.

   Plot previews can be enlarged and exported as PNG, SVG, or PDF. Selected
   raw arrays can be downloaded as TSV, TXT, CSV, or NPZ. The application
   Settings menu controls plot DPI and raw compression; GZIP is the default.
   Conversion memory limit (MB) sets the estimated working-memory budget for
   BigWig and bedGraph preparation, from 64 MB (default) to 4096 MB. Choosing
   512 MB allows the K562 example at 1 kb bins for all chromosomes (about
   322 MB estimated). Related track array and download limits scale with the
   budget. It does not reserve RAM, cap actual process RAM use, or change upload
   limits. Settings uses grouped switches and draft preferences; **Apply** saves
   changes. **Restore defaults** fills the draft without clearing data or results.
   Under **Interface**, **Field hover hints** controls short descriptions shown
   when pointing at fields. It is on by default and can be disabled independently
   of plot hover readouts. **Field help** remains a separate, more detailed option
   beside fields and is off by default.
   Under **Visualisation**, **DNAscape theme** changes the accent colour across the
   interface, default plot curves, logos, and browser-tab icon. Choose a colour
   with the picker or enter a hex value; light and dark variants keep labels
   readable. **Restore defaults** restores the original DNAscape red.
   This section also contains **Controls font size** and **Preview points**.
   The point budget defaults to 1,000 per curve in the visible range, with
   100–10,000 supported. Applying it updates mounted previews without resetting
   their zoom window. Plot settings default to an aspect ratio of 2.5, edited
   beside **View** in **Style**. **Fill** shades below or above curves and KDEs
   using a light version of each curve's colour, within the displayed Y limits.
   Application restart and
   Reset everything restore the 64 MB default.
   Histogram views expose synchronized bin-count, exact bin-size, and editable
   X-range controls. Changing the range preserves the requested bin width and
   updates the count, while the reset action restores the full data range. Bars
   can touch or use a small visual gap, and the boundary selector determines
   whether values exactly on an internal edge are counted in ``[a, b)`` or
   ``(a, b]``; both outer range endpoints remain included. Histograms display
   the mean, median, and population standard deviation by default in a compact
   **Statistics** panel combined with the legend. The panel defaults to the
   top right and can be positioned in any corner. For combined histograms,
   choose a dataset from the panel dropdown or click its bins when Statistics
   is enabled. Turning Statistics off replaces the panel with a legend listing
   every dataset, without a selector. **Statistics** sits below Legend, Grid and
   Hover in Guides. **Export statistics**, beside it, independently adds a file
   to Raw data with each plotted dataset's mean, median and population standard
   deviation at full precision. Filename, TSV/TXT/CSV/NPZ format and compression
   remain editable, including for separate Analysis plots. Bulk raw-data downloads
   include the enabled statistics file. **Side by side**
   places datasets within each shared bin interval, so bin size still refers
   to the whole interval. **KDE** replaces bars with Gaussian density curves;
   **Smoothness** scales Scott's bandwidth, with 1 as the default and larger
   values producing smoother curves.

   Cartesian previews support dragging to pan within the default axis bounds. Hold
   left Ctrl and scroll to zoom X and Y together, or hold left Ctrl and drag
   horizontally to select a shaded region and zoom only X. Heatmaps instead
   select an X/Y box. The axis fields
   and plot downloads reflect the current view. Axis reset arrows restore
   the default ranges. Legends default to the top right; fit-to-window and
   download buttons appear at the bottom right above the X axis. Both axis
   lines remain visible while zoomed in. Zooming refines curves and heatmaps
   from the underlying data to reveal detail in the visible region. **Hover** in Guides
   toggles floating value readouts, dashed guides, and coloured tracking points
   on curves. Floating boxes contain only Y values, or Z for heatmaps; axis
   readouts use the DNAscape theme colour. Heatmaps show both X and Y guides.
   Clear an axis-label field to remove its plot title.
   **Circular** view locks the aspect ratio to 1. **Track height (%)**, down to
   1%, and **Inner radius (%)** control the tracks and central opening.
   **Combine** appears for multiple datasets and overlays them instead of using
   separate rings. A mean, its standard deviation and individual simulation
   profiles share one track and scale. Left Ctrl and scrolling magnifies the
   view; Left Ctrl and dragging magnifies the smaller arc while preserving its
   original angular position, including across zero. Normal dragging pans
   around the circle.
   Hovering inside the central opening also draws a dashed radial guide from
   the centre, one position readout beside the highlighted point, and matching curve points.
   Floating boxes contain only coloured signal values. The legend stays fixed
   in the selected corner during zooming. **Legend position** defaults to Top right
   and also offers Top left, Bottom right and Bottom left. Exported HTML uses
   the same navigation and hover controls.
   Intrinsic simulation distributions,
   including S-phase duration, origins per replicate and inter-origin or replicon
   lengths, use Histogram or Boxplot views instead.
   Logarithmic axes omit zero and negative samples and start at the smallest
   positive native value. Explicit zero or negative axis minima produce an
   error instead of silently changing the selected bounds.

Analysis
   Upload one or more profiles and create profile, heatmap, histogram, or
   box-plot previews with the same plotting controls used by Mapping and
   Simulation.

   Combined layouts offer per-dataset **Curves**, **Histograms**, or **Boxes**
   sections, depending on the view. All support editable names and colours.
   Histograms add opacity and, when KDE is on, thickness and type. Boxes add
   outline thickness, numeric X position, and an optional X label. Leaving the
   X label empty shows the position. Each box is centered on its tick, including
   after zooming. These sections are hidden when the layout is Separate.

   **Enable analysis** is off by default. Turning it on changes **Plot** to
   **Analyse** and reveals Spearman, Pearson, and Kendall tau controls.
   The **Analysis report** below the plotting settings combines the methods in
   one table, with a grouped heading for each correlation above its dataset
   columns. **Show p-values** adds two-sided, unadjusted p-values beneath the
   coefficients; self-correlations have no p-value. The export row has an
   editable report name, File type, a download arrow, and a cross to clear the
   preview while retaining uploaded data. Downloads default to CSV, with TSV
   also available, and include all coefficients, paired sample counts, and
   optional p-values. Disabling analysis hides its correlation controls and report.

   **Data transforms** and **Region transforms** follow Data Manager's controls.
   Both are collapsed and off by default, work independently of **Enable analysis**,
   and have a **Dataset** menu for one dataset or all datasets. Data transforms
   offers Python/NumPy formulas using **source**, or **source1**, **source2**, etc.
   with **Combine**, followed by smoothing, value scaling and inversion.
   Region transforms defines a mask and edits only its selected samples. The
   default replaces negative values with zero. **Keep original profiles** retains
   the profiles before the regional edit beside their transformed versions.

   **Analysis on mask only**, below the other Region transforms switches, limits
   the selected datasets' correlations and distribution statistics to masked
   samples. Comparisons use overlapping finite samples without shifting their
   indices. Curve plots and raw profile exports retain the complete arrays.
   **Export mask** adds a positions/intervals download in Raw data, with integer
   coordinates and TSV, TXT, CSV or NPZ formats. The exported Python script
   reproduces these settings. Every run starts from the uploads.

   Each selected dataset has independent Off, Moving average, or Gaussian
   smoothing controls. Smoothing applies to both plots and correlations on
   the next action, preserving the source values, sample positions, and missing
   samples. Time profiles are smoothed over time; space-time fields are smoothed
   over position at each time. Correlations pair matching indices over the shared
   span and exclude non-finite pairs. Space-time overlap is calculated along both
   axes before flattening. Constant profiles, fewer than two paired samples, and
   mismatched dependences have undefined coefficients, shown as a dash and
   exported as empty cells. Hovering a coefficient shows its paired sample count.

   X and Y limits remain editable in every view, with Z (colour) limits for
   heatmaps. Reset restores the automatic range, including after an incomplete
   edit. Heatmap ranges can crop or extend the displayed axes without changing
   the original data coordinates. Enabled grid lines appear over the heatmap.
   Large heatmaps are automatically averaged into at most 300 time rows by
   720 position columns using bounded working tiles; raw exports retain the
   original resolution. PDF and SVG exports embed only the heatmap as a PNG,
   keeping axes, grid lines, colour scales and labels as vectors.

Cached Files
   Search, sort, download, reopen, or remove uploads, prepared datasets,
   generated files, and completed workflows retained in the current browser
   tab. The oldest entries are removed automatically above 30 items per
   category, 80 items overall, or 256 MB retained memory. The cache is cleared
   when the app process ends.

Resources
   Open the repository, documentation, issue tracker, and citation guidance.

.. figure:: _static/img/app_overview_placeholder.svg
   :width: 760
   :alt: Placeholder for the NiceGUI overview page

   DNAscape application layout.

Constant Simulation Example
---------------------------

1. Open **Simulation**.
2. Keep both inputs set to **Constant**.
3. Choose the genomic interval, resolution, replicate count, and seed.
4. Leave both optional trackers off for the quickest summary preview. Enable
   **Track time courses** for replicated fraction, fork-count and origin-firing
   curves, and the completed replicon-length distribution; enable **Track
   space-time outputs** for initiation/coalescence and directional/total
   fork-density heatmaps. Inter-origin-distance histograms and box plots are
   always available.
5. Select **Simulate** and monitor progress in Preview.
6. Choose the output to display, then use the hover toolbar or Raw data section
   to export it.

Safety Limits
-------------

The local interface accepts up to 50,000 simulations per ensemble in both
Simulation and Mapping. It validates numeric inputs and genuine retained-array,
spatial-grid, temporal-grid, upload, and browser-payload limits before calling
``rsim``. It does not reject an otherwise valid run merely because the number
of bins multiplied by the number of simulations is large. Oversized retained
outputs are rejected with guidance to coarsen the resolution or shorten the
requested window.
