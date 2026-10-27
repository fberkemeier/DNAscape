Examples
========

Short recipes. For guided workflows, see :doc:`notebooks`.

Interface
---------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Goal
     - App
     - Python
   * - Inspect data
     - Upload, convert, plot.
     - Script quality control.
   * - Map features
     - Try routes quickly.
     - Save exact assumptions.
   * - Simulate
     - Compare a few settings.
     - Run sweeps or large ensembles.
   * - Make figures
     - Fast exports.
     - Full matplotlib control.

Timing To Initiation Rate
-------------------------

.. code-block:: python

   from dnascape import rload, rmap, rsave

   timing = rload(
       "data/replication_timing/replication_timing_H1.tsv",
       chromosome="chr1",
       data_form="space",
   )

   initiation_rate = rmap(
       source={"replication_timing": timing, "fork_speed": 1.4},
       target="initiation_rate",
       assumptions={"resolution_x": 1.0, "target_form": "space", "maxiter": 20},
   )

   rsave(initiation_rate, "outputs/H1_chr1_initiation_rate.tsv", data_form="space")

Import bedGraph
---------------

.. code-block:: python

   from dnascape import rtrack

   rfd = rtrack(
       "tracks/okseq.bedgraph",
       chromosome="chr1",
       start=0,
       end=25_000_000,
       bin_size=10_000,
       aggregation="mean",
       interpolation="linear",
       smooth_window=3,
       normalize="center",
       data_form="space",
   )

Perturbation Sweep
------------------

.. code-block:: python

   from dnascape import rsim

   simulations = {}
   for speed in [1.4, 1.0, 0.7]:
       simulations[speed] = rsim(
           source={"initiation_rate": initiation_rate, "fork_speed": speed},
           assumptions={
               "sim_number": 200,
               "resolution_x": 1.0,
               "time_statsQ": True,
               "seed": 100 + int(speed * 10),
               "verbose": False,
           },
       )

Compare Timing
--------------

.. code-block:: python

   import numpy as np
   from dnascape import plotf

   predicted = simulations[1.4]["replication_timing"]
   valid = np.isfinite(timing) & np.isfinite(predicted)
   mse = np.mean((timing[valid] - predicted[valid]) ** 2)

   fig, ax = plotf(
       timing,
       predicted,
       labels=["Observed", f"Predicted, MSE={mse:.2f}"],
       xtitle="Genomic position (kb)",
       ytitle="Replication timing (min)",
       invyQ=True,
   )

Use Cases
---------

* Timing to initiation-rate inference.
* Initiation profiles to predicted timing.
* RFD comparison with timing-derived directionality.
* Simulation of fork slowing, origin changes, stalling, or circular DNA.
* Benchmarking against simulated ground truth.

Tutorial Order
--------------

1. ``examples/0-quickstart.ipynb``
2. ``examples/1-data_management.ipynb``
3. ``examples/2-mapping.ipynb``
4. ``examples/3-simulation.ipynb``
5. ``examples/4-neat_examples.ipynb``
