API
===

Import public functions from ``dnascape``:

.. code-block:: python

   from dnascape import rmap, rsim, rload, rsave, rdata, rtrack

Mapping
-------

``rmap(source=None, target=None, data=None, assumptions=None, method="auto", return_info=False, **kwargs)``
   Map one replication feature to another. Use ``source={feature: data}`` or
   ``source="feature", data=...``.

Route helpers:

.. code-block:: python

   rmap.targets()
   rmap.requirements(target="firing_rate")
   rmap.requirements(target="replication_timing", detail=True)
   rmap.canonical("rt")

Simulation
----------

``rsim(..., source=None, assumptions=None)``
   Simulate stochastic S phases from firing rate and fork speed. Prefer
   ``rsim(source={...}, assumptions={...})`` for clear workflows.

Reference tables:

.. code-block:: python

   rsim.input()
   rsim.output()

Data
----

``rload(source, source_format="auto", data_form="auto", chromosome=None, value_column=None, ...)``
   Load DNAscape-ready data.

``rsave(data, output_path, target_format="auto", data_form="auto", ...)``
   Save arrays or tables.

``rdata(source, target="tsv", output_path=None, source_format="auto", data_form="auto", ...)``
   Convert files. BigWig and bedGraph inputs are binned through ``rtrack``.

``rtrack(source, source_format="auto", chromosome=None, start=None, end=None, bin_size=1000, ...)``
   Bin BigWig or bedGraph tracks.

``inspect_genomic_track(source, source_format="auto")``
   Summarize chromosomes and intervals in a track.

Plotting
--------

``plotf``
   Profile plots.

``mplotf``
   Multi-panel profile plots.

``ploth``
   Histograms.

``plotw``
   Whisker or violin summaries.

``plothm``
   Heatmaps.

Legacy Helpers
--------------

Existing notebooks may use ``loadcsv``, ``loadtxt``, ``savecsv``, ``savetxt``,
``select_profile_region``, ``smoothf``, ``rescale``, and
``mean_squared_error``. New workflows should usually prefer ``rload``,
``rsave``, ``rdata``, ``rtrack``, ``rmap``, and ``rsim``.

Features
--------

Canonical feature names include:

* ``firing_rate``
* ``fork_speed``
* ``kinetic_fields``
* ``replication_timing``
* ``fork_directionality``
* ``replicated_fraction`` / ``replicated_fraction_xt``
* ``fork_density`` / ``fork_densities_xt``
* ``initiation_events`` / ``initiation_events_xt``
* ``coalescence_events`` / ``coalescence_events_xt``
* ``origin_efficiency``
* ``inter_origin_distance``
* ``s_phase_duration``
* ``completion_time``
* ``num_origins``

Aliases such as ``rt``, ``rfd``, and ``fr`` are accepted, but canonical names
are clearer in saved workflows.
