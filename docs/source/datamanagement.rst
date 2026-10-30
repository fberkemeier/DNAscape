Data Manager
============


Use **Data Manager** to convert experimental or model-derived files and
genomic tracks into reusable profiles, with explicit coordinates, resolution
and value processing.

Inputs
--------

Use **Upload** for local data or **Link** for a public file URL. Click
**Inspect** after entering a link. A successful inspection becomes an inactive
check button; editing the link makes it inspectable again.

Common inputs are TSV, TXT, CSV, NPZ, BigWig and bedGraph. Inspection identifies
available arrays, columns, chromosomes and dependence. Each source's details
appear below its upload or link. For multiple sources, numbered source labels
keep configuration and formulas associated with the correct input.

See :ref:`concept-assays` for the meaning of Repli-seq, SNS-seq and OK-seq signals.

Main workflow
-------------

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

Apply any optional transforms below, then click **Convert**.

Optional settings
-----------------

Data transforms
~~~~~~~~~~~~~~~

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

Region transforms
~~~~~~~~~~~~~~~~~


Region transforms run after **Configure** and **Data transforms**. They retain
the grid and change only selected values. Use them, for example, to replace
negative values, interpolate a selected interval or modulate the signal with
another prepared profile.

The selection, operation and mask export controls are described under
:ref:`region-transforms`. Here ``source`` means the prepared signal, including
the result of a combined data-transform formula.

Outputs and interpretation
--------------------------

Click **Convert** to create a result. The latest file is selected at the top
of **Generated Files**; select another row to change the displayed result.
**Raw data** contains the generated files and any enabled mask exports.
Download them as TSV, TXT, CSV or NPZ, or use **Open in** to send a profile to
Mapping, Simulation or Analysis.

When an applied mask is available, **Plot settings -> Guides -> Mask** shades
it in light grey. With Legend enabled, name the signal and mask separately.
Mask shading uses the selection made before the regional edit.

The :doc:`quickstart` gives a concrete Repli-seq example.
Use :doc:`exportworkflows` to reproduce the current app settings in Python,
or :doc:`pythonmodules` for the direct Python functions.
