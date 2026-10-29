Data Manager: rdata
===================

``rdata`` prepares experimental or model-derived data for DNAscape. In the app,
**Data Manager** converts files and genomic tracks into reusable profiles,
with explicit coordinates, resolution and value processing.

Load a source
-------------

Use **Upload** for local data or **Link** for a public file URL. Click
**Inspect** after entering a link. A successful inspection becomes an inactive
check button; editing the link makes it inspectable again.

Common inputs are TSV, TXT, CSV, NPZ, BigWig and bedGraph. Inspection identifies
available arrays, columns, chromosomes and dependence. Each source's details
appear below its upload or link. For multiple sources, numbered source labels
keep configuration and formulas associated with the correct input.

Configure the profile
---------------------

**Apply configuration** is on by default. **Prepare as** describes the
biological quantity you intend to use. Select the appropriate chromosome,
column or array, then choose the region and bin size.

For BigWig and bedGraph, coordinates are **bp**. Aggregation controls how
interval signals contribute to each output bin. **Mean** is a useful starting
point for continuous signals; **Maximum** retains strong local peaks, while
**Coverage** measures occupied width. Choose a rule appropriate to the assay.

DNAscape-formatted spatial inputs also support **Start**, **End**, source
resolution and output bin size. Reset the region limits to restore the full
source. Wider bins average the available profile; finer grids interpolate.
The selected physical interval remains fixed.

In **Values**, set decimal places, missing-value handling and optional
**Lower bound** / **Upper bound**. Values outside these bounds become missing
and follow the selected missing-value policy. Bin positions are retained.
Keep missing values distinct from measured zero when that distinction matters.

With several sources, **Apply to all** shares Configure settings by default.
Turn it off to choose a numbered **Source** and configure inputs separately.
**Show plot**, beside Apply configuration, controls the preview.
Turning Apply configuration off bypasses its value processing; genomic tracks
still need a region and binning to become arrays.

Data transforms
---------------

Open **Data transforms** to change the prepared signal. Enable **Apply
transform** and enter a numeric Python/NumPy expression. The default
``source`` preserves the input. For example:

.. code-block:: python

   source / np.mean(source)
   np.log1p(source)
   np.where(source < 0, 0, source)

Each line above is an alternative formula, entered separately. Expressions
must preserve the array shape; a scalar result fills the source grid.
The supported numeric expressions do not execute general Python statements.

With multiple sources, **Combine** replaces the separate formulas with one
**Combined transform formula**, using ``source1``, ``source2`` and so on:

.. code-block:: python

   0.7 * source1 + 0.3 * source2

Combined sources must have matching shapes and supplied coordinates. The
result is one profile and one generated file. **Smoothing**, **Value scaling**
and **Invert** follow the formula within Data transforms. Invert remains
available independently of the selected scaling mode.

.. _data-region-transforms:

Region transforms and masks
---------------------------

Region transforms run after **Configure** and **Data transforms**. They retain
the grid and change only selected values. The section is collapsed and
**Apply transform** is off by default; its controls remain editable while off.

Under **Select regions / define mask**, choose:

* **Signal threshold**, or a threshold relative to the source minimum or maximum.
* **Between thresholds**, selecting values within or outside the interval.
* **Missing values (NaN)**, or **Whole region**.
* **Position ranges**, such as ``(1000,2000), (4000,5000)``.
* **Upload profile**, for a position list or two-column interval file.

Position ranges and intervals include their start and exclude their end.
**Coordinates** defaults to **bp**; choose kb if needed. **Invert mask**
selects the complement within the configured region. Empty position ranges or
an absent uploaded profile select nothing.

Choose a **Transform** operation: replace with a value or NaN, average each
selected region, interpolate between surrounding values, add Gaussian noise,
match a reference minimum/maximum, multiply or add a value. The default mask
selects negative values and replaces them with zero.

**Modulate with profile** uses a second prepared genomic signal or annotation.
Enter a formula such as ``source * profile``. Here ``source`` is the current
prepared signal, including any combined transform; ``profile`` is the uploaded
covariate. The formula can use surrounding data, but only masked values change.
For mismatched profiles, modulation aligns from the beginning and uses the
overlapping extent, leaving other source values unchanged and issuing a yellow
warning. Check alignment when interpreting genomic effects.

Enable **Export mask** beside Apply transform to add a mask file to Raw data.
**Export as** chooses a position list or merged intervals. Genomic mask exports
use integer **bp** coordinates; text exports have no header or extra columns.
These files can be reused in another page's Region transforms.

Convert, inspect and reuse
--------------------------

Click **Convert** to create a result. The latest file is selected at the top
of **Generated Files**; select another row to change the displayed result.
**Raw data** contains the generated files and any enabled mask exports.
Download them as TSV, TXT, CSV or NPZ, or use **Open in** to send a profile to
Mapping, Simulation or Analysis.

When an applied mask is available, **Plot settings -> Guides -> Mask** shades
it in light grey. With Legend enabled, name the signal and mask separately.
Mask shading uses the selection made before the regional edit.

Assay meaning
-------------

Repli-seq, SNS-seq and OK-seq measure different aspects of replication. Prepare
the quantity required by the next workflow:

* **Repli-seq:** a relative timing score needs an explicit time calibration
  before it represents minutes. For sixteen equally spaced chronological S-phase fractions,
  Data Manager also supports a weighted mean time or sigmoid T50 estimate
  using the specified S-phase duration.
* **SNS-seq and related origin-enrichment data:** preserve the spatial origin
  signal, then justify its scale before treating it as an initiation rate.
* **OK-seq:** preserve signed directionality and verify strand orientation.

The :doc:`quickstart` gives a concrete Repli-seq example. :doc:`python` shows
``rdata`` in a script, while :doc:`foundations` defines the model quantities.
