Troubleshooting
===============

Import Fails
------------

Install DNAscape in the active environment:

.. code-block:: bash

   python -m pip install -e .

Missing NumPy, pandas, matplotlib, seaborn, or SciPy usually means the package
was installed in a different environment.

App Fails
---------

Run from the repository root:

.. code-block:: bash

   python -m pip install -r app/requirements.txt
   python -m app

Missing Column
--------------

Check headers before loading:

.. code-block:: python

   import pandas as pd

   table = pd.read_csv("input.tsv", sep="\t")
   print(table.columns)

Then pass ``chromosome="chr1"`` or ``value_column="sample_name"``.

Shape Mismatch
--------------

Inputs mapped together must have the same number of genomic bins. If all
kinetic inputs are scalar or time-only, pass ``n_bins``, ``genome_length``, or
``interval_kb``.

Missing Values
--------------

Handle NaNs before mapping or simulation. For tracks, use ``missing``,
``interpolation``, ``smooth_window``, and ``normalize`` in ``rtrack``.

BigWig Fails
------------

Install ``pyBigWig``:

.. code-block:: bash

   pip install pyBigWig

BigWig import needs an explicit chromosome. For multiple tracks, set the same
``chromosome``, ``start``, ``end``, and ``bin_size``.

Mapping Is Slow
---------------

Use a smaller region, coarser ``resolution_x``, fewer iterations, or fewer
simulations. Start with constant fork speed.

Simulation Is Large
-------------------

Avoid ``time_stats_xtQ=True`` unless you need heatmaps or event densities.
Use coarser space/time resolution or ``dens_memmap_dir`` for large runs.

Results Change
--------------

Set ``seed``:

.. code-block:: python

   rsim(
       source={"firing_rate": firing_rate, "fork_speed": 1.4},
       assumptions={"sim_number": 100, "seed": 1, "verbose": False},
   )

App Or Package
--------------

Use the app for inspection and short runs. Use Python for reproducible
analysis, sweeps, large simulations, and publication pipelines.

Docs Build
----------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html docs/source docs/build/html

If a figure warning appears, check ``docs/source/_static/img``.

Placeholders
------------

Placeholder figures remain for app screenshots, mapped profiles, heatmaps,
simulation summaries, and observed-versus-predicted comparisons.
