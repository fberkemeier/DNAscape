Common controls
===============


The workflow pages share plot controls, file exports and region transforms.
Use this page for their common behaviour, then follow the relevant page in
the :doc:`appguide` for its inputs and model settings.

Navigation and running jobs
---------------------------


After :doc:`installation`, use the left menu to move between pages. A spinner
identifies a running job; a check marks a result completed while you were
elsewhere. Jobs can continue while you inspect another page. Use the page's
stop action to cancel its run.

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

Files, exports and cached results
---------------------------------

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

.. _region-transforms:

Region transforms and masks
---------------------------

A **mask** selects positions; a **region transform** specifies what happens
there. The controls are shared across the app, with a different role on each
page:

* **Data Manager:** edit prepared values after configuration and data transforms.
* **Mapping:** transform selected target values from the initial guess, then
  keep those values fixed during fitting.
* **Simulation:** modify the initiation-rate input, fork-speed input or both.
* **Analysis:** edit selected datasets and optionally restrict statistics to
  masked positions using **Analysis on mask only**.

**Apply transform** is off by default; the controls remain editable while
it is off. **Export mask** can be selected while the transform is off, but
an export is produced only when it is applied.

Under **Select regions / define mask**, choose:

* **Signal threshold**, or a threshold relative to the source minimum or maximum.
* **Between thresholds**, selecting values within or outside the interval.
* **Missing values (NaN)**, or **Whole region**.
* **Position ranges**, such as ``(1000,2000), (4000,5000)``.
* **Upload profile**, for a position list or two-column interval file.

Position ranges and intervals include their start and exclude their end.
**Coordinates** defaults to **bp**; choose kb if needed. **Invert mask**
selects the complement within the selected region. Empty position ranges or
an absent uploaded profile select nothing.

Choose a **Transform** operation: replace with a value or NaN, average each
selected region, interpolate between surrounding values, add Gaussian noise,
match a reference minimum/maximum, multiply or add a value. The default mask
selects negative values and replaces them with zero.

**Modulate with profile** uses a second prepared genomic signal or annotation.
Enter a formula such as ``source * profile``. Here ``source`` is the signal
being transformed; ``profile`` is the uploaded covariate. The formula can use
surrounding data, but only masked values change.
For mismatched profiles, modulation aligns from the beginning and uses the
overlapping extent, leaving other source values unchanged and issuing a yellow
warning. Check alignment when interpreting genomic effects.

Enable **Export mask** beside Apply transform to add a mask file to Raw data.
**Export as** chooses a position list or merged intervals. Genomic mask exports
use integer **bp** coordinates; text exports have no header or extra columns.
These files can be reused in another page's Region transforms.

Plot settings and navigation
----------------------------

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

Settings
--------

Open **Settings** from the app menu:

* **Interface:** retain controls and results with **Keep settings and results
  between pages**. **Field hover hints** controls short cursor explanations;
  the separate **Field help** option enables longer explanations.
* **Files and exports:** set file naming, compression, export defaults and
  the conversion memory budget.
* **Visualisation:** set theme, control font size and **Preview points**
  (1,000 per visible curve by default). Zooming requests more local detail.
* **Navigation controls:** a short reminder of the mouse and keyboard actions.

Export an app workflow
----------------------


Enable **Settings -> Interface -> Developer mode** to show **Code** on each
workflow page. See :doc:`exportworkflows` for the script, input placeholders
and output folders.
