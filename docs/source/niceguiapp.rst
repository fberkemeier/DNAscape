The app
========

Use the DNAscape app to prepare inputs, choose a model and inspect its outputs.
The main pages correspond to the three modules: **Data Manager (rdata)**,
**Mapping (rmap)** and **Simulation (rsim)**. **Analysis** compares prepared
profiles; **Cached Files** helps reuse earlier inputs and results.

.. _app-open-browser:

Open and navigate
-----------------

Start the app from the project folder with your virtual environment active:

.. code-block:: bash

   python -m app

Open the address printed in the terminal. To launch the browser automatically
in Windows PowerShell:

.. code-block:: powershell

   $env:DNASCAPE_APP_SHOW = "true"
   python -m app

On macOS or Linux:

.. code-block:: bash

   DNASCAPE_APP_SHOW=true python -m app

Use the left menu to move between pages. A spinner identifies a running job;
a check marks a result completed while you were elsewhere. Jobs can continue
while you inspect another page. Use the page's stop action to cancel its run.

Inputs, controls and results
----------------------------

Each workflow separates its input controls from a preview and downloadable
outputs. Start with a small region and simple model, then increase detail
when the inputs and units are correct. Controls appear when relevant to the
selected input, target and dependence.

**Whole region** uses the available spatial profile. Otherwise, **Start** and
**End** select a half-open interval. **Resolution** specifies bin width; it
changes the number of values over that interval. See :ref:`concept-grids` for
units and resampling.

**Show plot** controls the preview, without removing numerical outputs.
Previews use bounded samples for responsiveness; raw downloads retain the
result's numerical resolution.

Reuse and export
----------------

Under **Raw data**, edit the file name and choose TSV, TXT, CSV or NPZ. NPZ is
useful for large arrays and space-time fields. The download control saves the
selected file; compression settings are available where supported.

The **Open in** button between download and remove transfers a file to
**Data Manager**, **Mapping**, **Simulation** or **Analysis**, including the
current page. When several input fields are possible, choose the destination.
For Mapping, the dialog lets you choose the target, route and precise input.
Mask files go to the destination page's **Region transforms** upload.

**Cached Files** can reopen retained workflows and download earlier artifacts.
It is temporary storage; save important data and settings outside the app.

Region transforms
-----------------

A **mask** selects positions; a **region transform** specifies what happens
there. The controls are shared across the app, with a different role on each
page:

* **Data Manager:** edit prepared values after configuration and data transforms.
* **Mapping:** transform selected target values from the initial guess, then
  keep those values fixed during fitting.
* **Simulation:** modify the initiation-rate input, fork-speed input or both.
* **Analysis:** edit selected datasets and optionally restrict statistics to
  masked positions using **Analysis on mask only**.

**Apply transform** is off by default. **Export mask** can be selected while
it is off, but an export is produced only when the transform is applied.
See :ref:`data-region-transforms` for the shared selection and operation controls.

Preview and navigation
----------------------

Drag to pan within the data limits. Hold **left Ctrl + scroll** to zoom around
the pointer. **Left Ctrl + drag** selects an x interval, or an x-y box in a
heatmap. Restore limits using the reset controls beside the axis limits.

Circular profiles preserve genomic angular positions: Ctrl-drag enlarges the
smaller arc, including across zero. Circular display is independent of the
model's **Periodic** boundary condition and is unavailable for heatmaps and
intrinsic simulation distributions.

Use **Plot settings** to edit axis labels, limits and presentation. In
**Guides**, **Hover** controls cursor readouts. Heatmap **Z-axis label** edits
the colour-bar label. These display choices do not resample exported raw data.

Analysis and settings
---------------------

In **Analysis**, upload profiles and enable analysis to calculate Pearson,
Spearman or Kendall correlations. Pairs use overlapping finite samples; report
exports retain paired counts and optional p-values. **Data transforms** and
**Region transforms** can act on one dataset or all datasets. **Analysis on
mask only** restricts statistics while preserving complete profile coordinates.

Open **Settings** from the app menu:

* **Interface:** retain controls and results with **Keep settings and results
  between pages**. **Field hover hints** controls short cursor explanations;
  the separate **Field help** option enables longer explanations.
* **Files and exports:** set file naming, compression, export defaults and
  the conversion memory budget.
* **Visualisation:** set theme, control font size and **Preview points**
  (1,000 per visible curve by default). Zooming requests more local detail.
* **Navigation controls:** a short reminder of the mouse and keyboard actions.

From app settings to Python
---------------------------

Enable **Settings -> Interface -> Developer mode**, then use **Code** on
Data Manager, Mapping, Simulation or Analysis. It downloads a single editable
``.py`` script with the current workflow options.

Set its input and output folders, place any listed original source files in
the input folder, and run it with the same DNAscape version installed. Constants
and prepared arrays may be included directly; external uploads are listed in
the script. This is a convenient starting point for :doc:`python` workflows.
