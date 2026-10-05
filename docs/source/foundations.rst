Core Concepts
=============

DNAscape works with measured tracks, inferred kinetic fields, and simulated
outputs on regular space and time grids.

Shapes
------

.. list-table::
   :header-rows: 1
   :widths: 20 18 30 32

   * - Form
     - Shape
     - Meaning
     - Examples
   * - Space
     - ``(n,)``
     - One value per genomic bin.
     - Timing, firing rate, fork speed, RFD.
   * - Time
     - ``(1, T)``
     - One value per S-phase time bin.
     - Replicated fraction, fork counts.
   * - Space-time
     - ``(n, T)``
     - Position by time.
     - Heatmaps, event densities, fork densities.

``resolution_x`` is kb/bin. ``resolution_t`` is minutes/bin. BigWig and
bedGraph coordinates are read in base pairs and converted after binning.

Quantities
----------

``replication_timing``
   Average replication time per genomic bin.

``firing_rate``
   Origin initiation propensity.

``fork_speed``
   Fork progression speed, usually kb/min.

``fork_directionality``
   Net replication direction, comparable to OK-seq-like data.

``replicated_fraction``
   Fraction replicated through S phase.

``replicated_fraction_xt``
   Local replicated fraction across position and time.

``fork_density`` / ``fork_densities_xt``
   Active fork density, optionally split by direction.

``origin_efficiency``
   Fraction of simulations in which a bin fires before passive replication.

``initiation_events`` / ``coalescence_events``
   Realized initiation and fork-coalescence events.

``completion_time``
   Absolute simulated replication-finish time on the simulation clock.

``s_phase_duration``
   Time from the first realised origin initiation to replication completion.

rmap And rsim
-------------

``rmap`` maps between compatible features. ``rsim`` simulates stochastic
S phases from firing rate and fork speed. They chain naturally:

.. code-block:: python

   firing_rate = rmap(
       source={"replication_timing": timing, "fork_speed": 1.4},
       target="firing_rate",
       assumptions={"resolution_x": 1.0, "target_form": "space"},
   )

   simulation = rsim(
       source={"firing_rate": firing_rate, "fork_speed": 1.4},
       assumptions={"sim_number": 100, "seed": 1, "verbose": False},
   )

Units
-----

* Positions in examples are kb.
* BigWig/bedGraph ``start``, ``end``, and ``bin_size`` are base pairs.
* Times are minutes.
* Default geometry is linear. Use ``geometry="periodic"`` or ``perQ=True`` for
  circular regions.

Reproducibility
---------------

Record ``resolution_x``, ``resolution_t``, region, smoothing, normalization,
geometry, ``sim_number``, and ``seed``. Set ``seed`` for simulations used in
comparisons or figures.

Downstream Use
--------------

DNAscape outputs are arrays and tables. Export them with ``rsave`` or
``rdata`` for statistics, plotting, clustering, or comparison with other
tracks.
