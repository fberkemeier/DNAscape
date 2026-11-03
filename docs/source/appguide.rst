App guide
=========


Use the left-hand menu to move between workflows. Start with
:ref:`niceguiapp` for plots, files, navigation, region transforms and Settings.
The workflow subsections below follow the pages of the app.

* :ref:`datamanagement` prepares experimental or model-derived profiles.
* :ref:`mappingfunctions` infers a target from compatible inputs or predicts
  replication features from a model.
* :ref:`simulationengine` samples individual S phases and ensemble statistics.
* :ref:`analysis` compares prepared profiles and datasets.

For a first run with specific settings, follow :ref:`quickstart`.

.. _niceguiapp:

User interface
--------------


The workflow pages share plot controls, file exports and region transforms.
Use this section for their common behaviour, then follow the relevant workflow
subsection below for its inputs and model settings.

Navigation and running jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~


After :ref:`installation`, use the left menu to move between pages. A spinner
identifies a running job; a check marks a result completed while you were
elsewhere. Jobs can continue while you inspect another page. Use the page's
stop action to cancel its run.

Inputs, controls and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~


Enable **Settings -> Interface -> Developer mode** to show **Code** on each
workflow page. See :ref:`exportworkflows` for the script, input placeholders
and output folders.

.. _datamanagement:

Data Manager
------------


Use **Data Manager** to convert experimental or model-derived files and
genomic tracks into reusable profiles, with explicit coordinates, resolution
and value processing.

Inputs
~~~~~~

Use **Upload** for local data or **Link** for a public file URL. Click
**Inspect** after entering a link. A successful inspection becomes an inactive
check button; editing the link makes it inspectable again.

Common inputs are TSV, TXT, CSV, NPZ, BigWig and bedGraph. Inspection identifies
available arrays, columns, chromosomes and dependence. Each source's details
appear below its upload or link. For multiple sources, numbered source labels
keep configuration and formulas associated with the correct input.

See :ref:`concept-assays` for the meaning of Repli-seq, SNS-seq and OK-seq signals.

Main workflow
~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

Data transforms
^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^


Region transforms run after **Configure** and **Data transforms**. They retain
the grid and change only selected values. Use them, for example, to replace
negative values, interpolate a selected interval or modulate the signal with
another prepared profile.

The selection, operation and mask export controls are described under
:ref:`region-transforms`. Here ``source`` means the prepared signal, including
the result of a combined data-transform formula.

Outputs and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Click **Convert** to create a result. The latest file is selected at the top
of **Generated Files**; select another row to change the displayed result.
**Raw data** contains the generated files and any enabled mask exports.
Download them as TSV, TXT, CSV or NPZ, or use **Open in** to send a profile to
Mapping, Simulation or Analysis.

When an applied mask is available, **Plot settings -> Guides -> Mask** shades
it in light grey. With Legend enabled, name the signal and mask separately.
Mask shading uses the selection made before the regional edit.

The :ref:`quickstart` gives a concrete Repli-seq example.
Use :ref:`exportworkflows` to reproduce the current app settings in Python,
or :ref:`pythonmodules` for the direct Python functions.

.. _mappingfunctions:

Mapping
-------


Replicated fraction is a space-time field, displayed as a heatmap.

Use **Mapping** to infer a target from compatible inputs, or to predict
observables from initiation rate and fork speed under explicit kinetic
assumptions. The equations and their limits are explained in :doc:`foundations`.

Inputs
~~~~~~

Start with **Mapped output**, then choose **Dependence** and **Input factors**.
Available routes and input fields update to match those choices.

.. list-table:: Common questions
   :header-rows: 1
   :widths: 33 28 39

   * - Question
     - Target
     - Typical supplied inputs
   * - Where is origin firing favoured?
     - Initiation rate
     - Replication timing or replicated fraction, together with fork speed.
   * - How quickly do forks move?
     - Fork speed
     - Timing or replicated fraction, together with initiation rate; supported routes also use directionality.
   * - When will each position replicate?
     - Replication timing
     - Initiation rate and fork speed.
   * - Which way do forks travel?
     - Fork directionality
     - Timing and fork speed, or kinetic inputs through a timing prediction.
   * - How does replication progress?
     - Replicated fraction (space-time) or fork density
     - Initiation rate and fork speed.
   * - Where and when do events occur?
     - Initiation or coalescence density
     - Compatible kinetic fields or prepared model outputs.

**Constant**, **Space**, **Time** and **Space-time** describe how a fitted
kinetic quantity varies. A time-only rate or speed is shared across the region;
a space-time field can vary along both axes. Kinetic fits with time dependence
use a Weibull hazard-shaped component. Space-time fits can infer a different
amplitude and Weibull shape at each position. These are constrained effective
kinetics; timing alone does not uniquely determine an arbitrary temporal
history. The shared equations and identifiability limits are explained in
:ref:`kineticframework`.

For initiation and coalescence outputs, **Quantity** distinguishes integrated
spatial density in events/kb from space-time rate density in events/(kb min).
These are actual event densities, distinct from potential initiation rate.
Use :ref:`simulationengine` for distributions across stochastic S phases.

Profiles and region
^^^^^^^^^^^^^^^^^^^

Under **Inputs**, enter constants with **Value**, or upload a profile and
select its dependence. Use the same chromosome, coordinate interpretation and
time reference for all inputs. Prepare unfamiliar track formats in Data
Manager first, where their region and value processing are easier to inspect.

In **Region**:

* **Whole region** uses the available spatial input. Otherwise set **Start**
  and **End** in kb; End is excluded.
* **Resolution** changes the bin width without stretching the chosen interval.
  Wider bins average the source; finer bins interpolate it.
* When a temporal grid is needed, its resolution controls time sampling.
  Automatic forward maps choose a time horizon that captures replication
  completion, subject to the supported model and resource limits.
* **Periodic** joins the ends. Use it for a periodic model, rather than to hide
  boundary effects in a linear region. Periodic fits use wrapped spatial context.
* **Show plot** controls the preview and its Plot settings.

Plain spatial uploads default to one sample per kb when no coordinate or
resolution metadata are available. Recheck the inferred interval before fitting.
The app keeps fork speed in kb/min and handles conversion to its internal grid.

Main workflow
~~~~~~~~~~~~~


Choose the target and input factors, supply the inputs, and define the region.
Start with the route's default fitting settings. Apply optional bounds or
region transforms when required, then run the map.

Click **Map**. The status line reports the stage, fit error when available,
ETA and elapsed time. **Initial prediction** measures the first forward
calculation; the estimate can change as later fitting stages are measured.
After that prediction the status shows completed/total fitting iterations.
The total covers all refinement grids, with line-search trials excluded.
Refinement and preview preparation retain the estimate instead of restarting it.

Optional settings
~~~~~~~~~~~~~~~~~

Fit settings
^^^^^^^^^^^^

Start with the route defaults. Change one control at a time and compare the
resulting prediction, not just the fitted curve.

**MSE threshold**
   Optional early stopping based on the discrepancy between observed and
   predicted data. It differs from a numerical floor on the target values.

**Fit iterations and Fit step**
   Spatial and temporal phases each default to **10 iterations**. Space-time
   fits expose **Space fit iterations** and **Time fit iterations**, each with
   its own step. **Fitting order** alternates the two phases, or completes space
   then time or time then space. One iteration is one proposed block update;
   line-search evaluations do not inflate the displayed counter. Fit steps
   limit changes in positive log-parameters, and rejected proposals retain the
   best prediction.

**Adaptive fitting and Fit tolerance**
   Large fits start on coarser grids and refine at the selected resolution.
   Iteration budgets are shared across grids; the reported final MSE uses all
   valid native observations. Disable Adaptive fitting for every update to
   use the selected grid. Fit tolerance defaults to 0.001: 0.1% timing error,
   or 0.001 fraction error. Zero disables convergence-based local skipping.

**Influence radius**
   Limits the distance of potential origins contributing to a position, in kb;
   the default is **2000 kb**. This is an explicit finite-influence
   approximation when physical cones extend farther. Increase it to check that
   a result is stable. Linear regions have no incoming forks from outside the
   supplied region; retain surrounding data when boundary behaviour matters.

**Weibull shape and Fit Weibull shape**
   Set the initial temporal shape and whether it is fitted. Shape 1 gives a
   constant hazard; shape 2 gives linear growth after the activation delay,
   while shapes below 1 permit decreasing profiles.
   Space-time fits can optimise the shape locally. Fork speed also has a
   positive basal component. **Onset interval** regularises the first part of
   the profile in physical minutes, independently of numerical resolution.
   The temporal profile describes prescribed kinetics, rather than drawing
   random Weibull fork speeds.

**Target bounds and Initial guess**
   Constrain the fitted quantity and choose its starting value. Supplied inputs
   remain fixed. Bounds apply to the actual fitted kinetic values on the
   computational grid; regional transforms can fix selected values separately.

The advanced **Fork movement** controls include stalling and, where needed,
the simulations averaged for a stochastic fit estimate. Deterministic routes
use the kinetic relationships directly; increasing simulations is not a
universal way to improve a mapping.

Region transforms
^^^^^^^^^^^^^^^^^

Open **Region transforms** and use **Relative to** to choose the supplied
input that defines the mask. Position ranges or **Upload profile** define
positions directly and do not need a signal reference. The shared controls
are described under :ref:`region-transforms`.

The transform is applied to the selected values of the initial target guess.
Those values are then held fixed during fitting, rather than multiplied,
perturbed or re-randomised at every step. Unmasked values continue to fit.
Observed timing and other supplied inputs remain unchanged.

A genomic mask on a constant or time-only target can require spatial
dependence. Mapping automatically selects a compatible target when available.
Transforms must retain physically valid target values: initiation rates are
non-negative and fork speeds are positive.

Outputs and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~

For inverse timing fits, inspect observed-versus-fitted timing, residuals and
MSE alongside the inferred target. A lower MSE measures agreement with that
objective; it does not make the inferred kinetics unique.

**Preview** can display inputs and target separately or together. Download
the fitted quantity and useful comparison data from **Raw data**, or transfer
a profile with **Open in**. Dependence suffixes ``_c``, ``_x``, ``_t`` and
``_xt`` distinguish constant, spatial, temporal and space-time map outputs.
Save the region, resolution and assumptions with the files.

The :ref:`quickstart` walks through timing-to-initiation mapping.
Use :ref:`exportworkflows` to reproduce all current app options in a script.

.. _simulationengine:

Simulation
----------


Use **Simulation** to generate stochastic S phases from potential initiation
rate and fork speed. Explore individual replication programmes, ensemble
variability and mechanisms beyond a deterministic mapping.

Inputs
~~~~~~

Supply **Initiation rate** and **Fork speed** as constants or profiles.
Profiles can depend on space, time or both, where supported by the input
controls. Initiation describes potential origin firing before passive
replication; it is not the number of origins that actually fire.

The app uses initiation rate **per spatial bin per minute** and fork speed
**in kb/min**. A mapped app profile can be transferred directly. Python users
passing a density returned by ``rmap`` must convert it to a per-bin rate;
see :ref:`concept-units` and :ref:`pythonmodules`.

Region
^^^^^^

**Start**, **End** and **Resolution (kb/bin)** define the simulated grid.
Enable **Whole region** to use a spatial input's full interval; this switch
is available when at least one kinetic input has a spatial profile. Otherwise,
enter the region explicitly. Uploaded values are resampled onto the selected
grid without changing its physical extent.

**Periodic** wraps forks across the region boundary; the default linear model
has two ends. **Show plot** controls preview generation. A circular display
is a presentation choice and does not enable periodic simulation.

Main workflow
~~~~~~~~~~~~~

**Simulations (replicates)**
   Number of independent S phases to average. Begin with a small ensemble while
   configuring the model; increase it to characterise population variation.

**Fix seed**
   Repeat a run with the same random seed and settings. Record both for
   reproducibility.

**Time window (min)**
   Interval covered by the recorded time-dependent outputs.

**Maximum time (min)**
   Stopping limit for a replicate. A recording window and a simulation stopping
   limit serve different purposes. Check for unfinished replication before
   interpreting completion statistics.

**Track time courses**
   Adds replicated fraction, origin firing, fork counts and available firing
   factors over time, and the completed replicon-length distribution.

**Track space-time outputs**
   Adds position-by-time matrices for replicated fraction, initiation and
   coalescence events, and directional/total fork density. These need more
   memory than profiles and time courses.

Choose time resolution fine enough for the dynamics of interest. More preview
points improve on-screen detail, not the simulation grid or stored outputs.

Optional settings
~~~~~~~~~~~~~~~~~

Fork movement and firing factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Fork stalling** enables genomic barriers at the specified **Stall rate
(stalls/kb)**. **Stall recovery** allows passage after a mean **Recovery time**
in minutes. With recovery disabled, a stalled fork remains blocked in that
realisation.

**Shared between directions** uses the same stall positions and recovery
landscape for forks arriving from either direction. With it off, the two
directions have independent landscapes. Shared genomic stalls do not pause
a distant sister fork.

**Stochastic movement** adds a positive speed distribution with the supplied
local speed as its arithmetic mean and the selected standard deviation in
kb/min. The model uses lognormal draws; zero deviation recovers deterministic
movement. Spatial and temporal speed dependence can be combined with stalling.

Finite firing factors provide an optional limited pool for origin activation.
Set its size and recycling behaviour when investigating resource-limited
initiation. Available-factor time courses require time tracking.

Region transforms
^^^^^^^^^^^^^^^^^

In **Region transforms**, set **Relative to** to **Initiation rate**,
**Fork speed** or **Both**. Each input has its own mask and operation when
both are selected. The controls match :ref:`region-transforms`.

These transforms modify the simulation inputs after region selection and
resampling. For example, set initiation to zero in a genomic interval to
suppress local origin firing while allowing incoming forks to replicate it.
A fork-speed transform must leave speeds finite and positive.

Outputs and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Click **Simulate**, then choose an output in **Preview**.

.. list-table:: Output groups
   :header-rows: 1
   :widths: 28 72

   * - Group
     - Interpretation
   * - Replication timing
     - Ensemble mean and standard deviation, with retained individual profiles.
   * - Origin efficiency
     - Fraction of replicates in which each bin fired before passive replication.
   * - Fork directionality
     - Mean replication direction along the region.
   * - Completion summaries
     - Completion times, S-phase durations and origins per replicate.
   * - Length distributions
     - Inter-origin distances, plus completed replicon lengths with time tracking.
   * - Outputs marked (time)
     - Recorded progression, actual origin firing, fork counts and firing factors.
   * - Outputs marked (space-time)
     - Position-by-time fields of progression, events and fork density.

**Origin firing (time)** counts actual initiations per recording interval,
averaged over replicates. It differs from the input initiation hazard and is
not an event rate per minute unless divided by interval duration.

**Completion time** is measured on the simulation clock. **S-phase duration**
is completion minus the first realised initiation. Inter-origin distance is
the separation of neighbouring fired origins; replicon length is the completed
DNA domain assigned to a fired origin. Unfinished domains are excluded from
the latter distribution.

Simulation event matrices contain averaged counts in recorded space-time
bins. Continuous rate densities produced by Mapping have a different
normalisation; see :ref:`concept-events` before comparing them.

Export and compare
^^^^^^^^^^^^^^^^^^

Download numerical outputs from **Raw data**, using NPZ for large matrices.
Keep the seed, replicate count, coordinates, input profiles and all mechanism
settings with the results. Use **Open in -> Analysis** to compare runs.

Mean curves can hide differences between individual S phases. Inspect
variability and distributions as well as means. For larger runs, request only
the time/space-time trackers and retained samples you need. Python exposes
additional retention controls in the :ref:`python-notebooks` examples.

.. _analysis:

Analysis
--------
