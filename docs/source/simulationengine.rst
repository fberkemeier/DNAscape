Simulation: rsim
================

``rsim`` generates stochastic S phases from potential initiation rate and fork
speed. Use **Simulation** to explore individual replication programmes,
ensemble variability and mechanisms beyond a deterministic mapping.

Set the kinetic inputs
----------------------

Supply **Initiation rate** and **Fork speed** as constants or profiles.
Profiles can depend on space, time or both, where supported by the input
controls. Initiation describes potential origin firing before passive
replication; it is not the number of origins that actually fire.

The app uses initiation rate **per spatial bin per minute** and fork speed
**in kb/min**. A mapped app profile can be transferred directly. Python users
passing a density returned by ``rmap`` must convert it to a per-bin rate;
see :ref:`concept-units` and :doc:`python`.

Define the region
-----------------

**Start**, **End** and **Resolution (kb/bin)** define the simulated grid.
Enable **Whole region** to use a spatial input's full interval; this switch
is available when at least one kinetic input has a spatial profile. Otherwise,
enter the region explicitly. Uploaded values are resampled onto the selected
grid without changing its physical extent.

**Periodic** wraps forks across the region boundary; the default linear model
has two ends. **Show plot** controls preview generation. A circular display
is a presentation choice and does not enable periodic simulation.

Choose ensemble and recording settings
--------------------------------------

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

Fork movement and firing factors
--------------------------------

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

Transform an input region
-------------------------

In **Region transforms**, set **Relative to** to **Initiation rate**,
**Fork speed** or **Both**. Each input has its own mask and operation when
both are selected. The controls match :ref:`data-region-transforms`.

These transforms modify the simulation inputs after region selection and
resampling. For example, set initiation to zero in a genomic interval to
suppress local origin firing while allowing incoming forks to replicate it.
A fork-speed transform must leave speeds finite and positive.

Read the outputs
----------------

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
------------------

Download numerical outputs from **Raw data**, using NPZ for large matrices.
Keep the seed, replicate count, coordinates, input profiles and all mechanism
settings with the results. Use **Open in -> Analysis** to compare runs.

Mean curves can hide differences between individual S phases. Inspect
variability and distributions as well as means. For larger runs, request only
the time/space-time trackers and retained samples you need. Python exposes
additional retention controls in the :ref:`python-notebooks` examples.
