Simulation with rsim
====================

``rsim`` simulates stochastic S phases from firing rate and fork speed.

Call
----

.. code-block:: python

   import numpy as np
   from dnascape import rsim

   simulation = rsim(
       source={"firing_rate": firing_rate, "fork_speed": 1.4},
       assumptions={
           "sim_number": 100,
           "resolution_x": 1.0,
           "resolution_t": 2.0,
           "time_grid": np.arange(0, 300 + 2, 2),
           "time_statsQ": True,
           "seed": 42,
           "verbose": False,
       },
   )

Inputs
------

``firing_rate`` and ``fork_speed`` can be:

* scalar;
* spatial ``(n,)``;
* time-only ``(1, T)``;
* space-time ``(n, T)``.

If both inputs lack a spatial axis, pass ``n_bins``, ``genome_length``, or
``interval_kb``.

Outputs
-------

Always returned:

* ``replication_timing``
* ``rep_times_per_sim``
* ``completion_time`` / ``s_phase_duration``
* ``num_origins`` / ``num_oris``
* ``inter_origin_distance`` / ``inter_origin_distances``
* ``origin_efficiency`` / ``efficiency``
* ``fork_directionality``

With ``time_statsQ=True``:

* ``time_grid``
* ``replicated_fraction``
* ``total_forks``
* ``active_forks``
* ``stalled_forks``
* ``firing_factors`` when finite factors are enabled.

With ``time_stats_xtQ=True``:

* ``replicated_fraction_xt``
* ``initiation_events_xt``
* ``coalescence_events_xt``
* ``right_fork_density_xt``
* ``left_fork_density_xt``

.. figure:: _static/img/simulation_outputs_placeholder.svg
   :width: 760
   :alt: Placeholder for simulation outputs

   Placeholder for simulation outputs.

Perturbations
-------------

Fork slowing:

.. code-block:: python

   slow_fork = rsim(
       source={"firing_rate": firing_rate, "fork_speed": 0.8},
       assumptions={"sim_number": 200, "resolution_x": 1.0, "seed": 3, "verbose": False},
   )

Fork stalling:

.. code-block:: python

   stalled = rsim(
       source={"firing_rate": firing_rate, "fork_speed": 1.4},
       assumptions={
           "sim_number": 200,
           "resolution_x": 1.0,
           "stall_rate": 0.02,
           "tau": 15.0,
           "seed": 4,
           "verbose": False,
       },
   )

Finite factors:

.. code-block:: python

   finite_factors = rsim(
       source={"firing_rate": firing_rate, "fork_speed": 1.4},
       assumptions={
           "sim_number": 200,
           "ffiring_nr": 60,
           "ffiring_recycle": 0.05,
           "time_statsQ": True,
           "seed": 5,
           "verbose": False,
       },
   )

Space-Time
----------

Space-time fields scale with bins times time points. Request them only when
needed.

.. code-block:: python

   heatmap_sim = rsim(
       source={"firing_rate": firing_rate, "fork_speed": 1.4},
       assumptions={
           "sim_number": 50,
           "resolution_x": 1.0,
           "resolution_t": 2.0,
           "time_grid": np.arange(0, 300 + 2, 2),
           "time_statsQ": True,
           "time_stats_xtQ": True,
           "seed": 8,
           "verbose": False,
       },
   )

.. figure:: _static/img/replicated_fraction_heatmap_placeholder.svg
   :width: 760
   :alt: Placeholder for replicated-fraction heatmap

   Placeholder for a replicated-fraction heatmap.

Compare
-------

.. code-block:: python

   import numpy as np

   predicted = simulation["replication_timing"]
   valid = np.isfinite(timing) & np.isfinite(predicted)
   timing_mse = np.mean((timing[valid] - predicted[valid]) ** 2)

Export:

.. code-block:: python

   from dnascape import rsave

   rsave(predicted, "outputs/simulated_timing.tsv", data_form="space")
   rsave(simulation["origin_efficiency"], "outputs/origin_efficiency.tsv", data_form="space")

App Page
--------

Simulation runs ``rsim`` interactively.

Inputs
   Firing rate and fork speed as constants or uploaded profiles.

Controls
   Region, resolution, replicates, geometry, time window, stalling, finite
   factors, seed, output, plot view.

Outputs
   Timing, efficiency, directionality, replicated fraction, fork counts,
   factors, and space-time fields.

Python
   ``rsim(source=..., assumptions=...)``.

Performance
-----------

* Use small ``sim_number`` while exploring.
* Avoid ``time_stats_xtQ=True`` unless needed.
* Coarser ``resolution_x`` and ``resolution_t`` reduce cost.
* Use ``dens_memmap_dir`` for large space-time outputs.
* Set ``seed`` for comparisons.
