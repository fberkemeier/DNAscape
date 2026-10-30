Mapping
========


Use **Mapping** to infer a target from compatible inputs, or to predict
observables from initiation rate and fork speed under explicit kinetic
assumptions. The equations and their limits are explained in :doc:`foundations`.

Inputs
--------

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
     - Replication timing and fork speed.
   * - How quickly do forks move?
     - Fork speed
     - Timing and initiation rate; supported routes also use directionality.
   * - When will each position replicate?
     - Replication timing
     - Initiation rate and fork speed.
   * - Which way do forks travel?
     - Fork directionality
     - Timing and fork speed, or kinetic inputs through a timing prediction.
   * - How does replication progress?
     - Replicated fraction or fork density
     - Initiation rate and fork speed.
   * - Where and when do events occur?
     - Initiation or coalescence density
     - Compatible kinetic fields or prepared model outputs.

**Constant**, **Space**, **Time** and **Space-time** describe how a fitted
kinetic quantity varies. A time-only rate or speed is shared across the region;
a space-time field can vary along both axes. Some routes offer a **Target
shape**, such as a Weibull time dependence.

For initiation and coalescence outputs, **Quantity** distinguishes integrated
spatial density in events/kb from space-time rate density in events/(kb min).
These are actual event densities, distinct from potential initiation rate.
Use :doc:`simulationengine` for distributions across stochastic S phases.

Profiles and region
~~~~~~~~~~~~~~~~~~~

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
-------------


Choose the target and input factors, supply the inputs, and define the region.
Start with the route's default fitting settings. Apply optional bounds or
region transforms when required, then run the map.

Click **Map**. The status line reports the stage, fit error when available,
ETA and elapsed time. **Initial prediction** measures the first forward
calculation; the estimate can change as later fitting stages are measured.

Optional settings
-----------------

Fit settings
~~~~~~~~~~~~

Start with the route defaults. Change one control at a time and compare the
resulting prediction, not just the fitted curve.

**MSE threshold**
   Optional early stopping based on the discrepancy between observed and
   predicted data. It differs from a numerical floor on the target values.

**Fit iterations and Fit step**
   Control the relevant fitting stage. Routes that separate spatial and
   temporal fitting expose **Space fit iterations** and **Time fit iterations**
   with their corresponding steps. Space-time fits may use sequential or
   alternating component updates; they can report more than one stage.

**Neighbourhood range and Maximum edge context**
   Control the spatial context used by supported calculations and linear-edge
   treatment. Cropped regions can lose contributions from origins outside the
   interval; retain surrounding data when local boundary behaviour matters.

**Target bounds and Initial guess**
   Constrain the fitted quantity and choose its starting value. Supplied inputs
   remain fixed. For a Weibull model, amplitude bounds apply to its fitted
   amplitude, whose units depend on the temporal shape.

The advanced **Fork movement** controls include stalling and, where needed,
the simulations averaged for a stochastic fit estimate. Deterministic routes
use the kinetic relationships directly; increasing simulations is not a
universal way to improve a mapping.

Region transforms
~~~~~~~~~~~~~~~~~

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
--------------------------

For inverse timing fits, inspect observed-versus-fitted timing, residuals and
MSE alongside the inferred target. A lower MSE measures agreement with that
objective; it does not make the inferred kinetics unique.

**Preview** can display inputs and target separately or together. Download
the fitted quantity and useful comparison data from **Raw data**, or transfer
a profile with **Open in**. Dependence suffixes ``_c``, ``_x``, ``_t`` and
``_xt`` distinguish constant, spatial, temporal and space-time map outputs.
Save the region, resolution and assumptions with the files.

The :doc:`quickstart` walks through timing-to-initiation mapping.
Use :doc:`exportworkflows` to reproduce all current app options in a script.
