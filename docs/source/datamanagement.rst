Data
====

Prepare data once, then reuse it in the package, app, and notebooks.

Forms
-----

.. list-table::
   :header-rows: 1
   :widths: 20 18 30 32

   * - Form
     - Shape
     - Use
     - Example
   * - ``space``
     - ``(n,)``
     - Genomic profile.
     - One chromosome column.
   * - ``time``
     - ``(1, T)``
     - S-phase time course.
     - Replicated fraction.
   * - ``spacetime``
     - ``(n, T)``
     - Position by time.
     - Heatmap or event density.
   * - ``table``
     - pandas table
     - Coordinates or labels must be kept.
     - ``chrom``, ``start_bp``, ``end_bp``, ``signal``.

Main helpers:

* ``rload``: load DNAscape-ready files.
* ``rsave``: save arrays or tables.
* ``rdata``: convert formats.
* ``rtrack``: bin BigWig or bedGraph tracks.

Files
-----

Supported input formats: CSV, TSV, TXT, NPY, NPV, NPZ, and gzipped CSV/TSV/TXT.
Export formats: TSV, TXT, CSV, and NPZ.

Load one chromosome:

.. code-block:: python

   from dnascape import rload

   timing = rload(
       "data/replication_timing/replication_timing_H1.tsv",
       chromosome="chr1",
       data_form="space",
   )

Convert a table:

.. code-block:: python

   from dnascape import rdata

   rdata(
       "input/repliseq.csv",
       "prepared/repliseq_chr1.tsv",
       chromosome="chr1",
       data_form="space",
       decimal_places=4,
   )

Save arrays:

.. code-block:: python

   from dnascape import rsave

   rsave(firing_rate, "outputs/firing_rate_chr1.tsv", data_form="space")
   rsave(replicated_fraction, "outputs/replicated_fraction.tsv", data_form="time")

Tracks
------

Use ``rtrack`` for BigWig and bedGraph. Track coordinates are base pairs;
DNAscape axes are usually kb.

.. code-block:: python

   from dnascape import rtrack

   timing = rtrack(
       "tracks/repliseq_chr1.bw",
       chromosome="chr1",
       start=0,
       end=10_000_000,
       bin_size=10_000,
       aggregation="mean",
       data_form="space",
   )

bedGraph:

.. code-block:: python

   rfd = rtrack(
       "tracks/okseq_chr1.bedgraph",
       chromosome="chr1",
       bin_size=5_000,
       aggregation="mean",
       interpolation="linear",
       smooth_window=5,
       normalize="center",
       data_form="space",
   )

Multiple tracks become a matrix when their bins match:

.. code-block:: python

   heatmap = rtrack(
       ["early.bw", "mid.bw", "late.bw"],
       chromosome="chr2",
       start=20_000_000,
       end=30_000_000,
       bin_size=25_000,
       labels=["early", "mid", "late"],
       data_form="spacetime",
   )

.. figure:: _static/img/replicated_fraction_heatmap_placeholder.svg
   :width: 760
   :alt: Placeholder for a space-time matrix

   Placeholder for a space-time matrix.

Tables
------

Wide chromosome tables are supported:

.. code-block:: text

   bin   chr1    chr2    chr3
   0     42.1    38.7    51.2
   1     41.9    39.2    50.8

Use ``chromosome="chr1"`` or ``value_column="sample_a"`` to select one
column.

Resolution
----------

* Use the same ``resolution_x`` for profiles mapped together.
* Small bins keep local structure but cost more.
* Large bins reduce noise and runtime.
* For tracks, ``resolution_x = bin_size / 1000``.

Cleanup
-------

``rtrack`` supports:

* ``missing="nan"``, ``"zero"``, or ``"fill"``;
* ``interpolation="linear"``, ``"ffill"``, or ``"bfill"``;
* ``smooth_window`` for rolling mean smoothing;
* ``normalize="minmax"``, ``"zscore"``, ``"center"``, or ``"maxabs"``.

Use the lightest cleanup that makes the assay usable, and record it.
Zero and custom-value fills are applied after smoothing and normalization.
Missing bins retain their positions; dropping them is rejected. Text and CSV
loaders preserve empty rows, leading bins, and internal gaps. Only trailing
padding in wide chromosome tables is cropped when selecting a chromosome.
The Data Manager defaults to filling empty bins with zero and applies fills
after value scaling. Mapping reports remaining missing values by default.

App Page
--------

Data Management converts DNAscape-ready files and genomic tracks.

Inputs
   CSV, TSV, TXT, NPV, NPY, NPZ, gzipped plain files, BigWig, and bedGraph.

Controls
   Target format, gzip, data form, column, NPZ key, rounding, chromosome,
   interval, bin size, aggregation, missing values, smoothing, normalization.
   Each source retains its own Configure settings. With multiple sources, use
   the numbered **Source** menu in Configure to select the one to edit; the same
   numbers appear above uploaded files and in public-link field labels. An inspected
   link shows a check mark in place of **Inspect**; editing it restores **Inspect**.
   **Apply configuration**
   is on by default. Turn it off to preserve values, missing samples and precision
   while retaining the track region and binning required for conversion.
   **Data transforms**, collapsed and off by default, runs after Configure and
   before Filter. Enable **Apply transform** to edit one Python/NumPy formula per
   source. With several sources, **Combine** replaces these fields with one
   formula using **source1**, **source2**, etc., and produces one plot and file.
   Array sizes and supplied coordinates must match. Formulas retain the source
   grid; scalar results fill it. NumPy arithmetic, conditional expressions,
   smoothing and convolution are available.
   The collapsed **Filter** section below Configure is off by default. Enable
   it to select regions by signal thresholds (including within or outside an
   interval), missing values, an uploaded mask
   or genomic positions/intervals, or the whole region. Its default replaces
   converted values below zero with zero. Operations include region means,
   interpolation, replacement with a value or NaN, Gaussian noise, reference
   minimum/maximum, multiplication, and addition. Filtering follows Configure
   and Data transforms, preserving bin positions; previews and downloads contain the edited data.
   Filter options remain editable with **Apply filter** off. In **Plot settings →
   Guides**, **Filter** highlights the applied filter's selected regions in light
   grey, using the selection before any filter edits. With **Legend** enabled,
   name the signal and filter separately. Shading follows zoom detail and does
   not change the exported data. Line previews refresh up to 1,000 native points by default
   from the visible window after zooming or panning.
   **Export filter**, next to **Filter** in Guides, adds a raw-data download of
   the selected positions before edits, independently of preview sampling.
   Text exports contain one integer per line with no header; NPZ contains one
   integer array. Genomic coordinates use base pairs; plain arrays use zero-based
   bin indices. Repeated positions form one selection across chromosomes or time slices.
   **Modulate with profile** combines a prepared signal or annotation profile
   through an editable Python/NumPy expression, defaulting to ``source * profile``.
   Here **source** is the current transformed or combined signal and **profile**
   is the uploaded modulation data. Both arrays must have matching dimensions.
   Formulas may use the whole profile for context, while only selected regions
   are replaced. Use **np.where** for conditional replacements and **np.nan_to_num**
   to handle non-finite values explicitly.

Outputs
   Converted file, inferred shape, track preview, download, cached artifact.

Python
   Use ``rdata`` for conversion and ``rtrack`` for track binning.

Export
------

Use TSV/CSV for readable profiles, NPZ for exact numerical intermediates,
and TXT for simple vectors.

Checklist
---------

Before mapping or simulation:

* same chromosome, region, and resolution;
* coordinate units recorded;
* missing values handled;
* raw and prepared files separated;
* exported names include feature, chromosome, and resolution.
