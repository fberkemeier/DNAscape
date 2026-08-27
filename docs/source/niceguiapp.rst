NiceGUI App
===========

Run the local interface from the repository root:

.. code-block:: bash

   python -m pip install -r app/requirements.txt
   python -m app

Open ``http://127.0.0.1:8080`` after the server starts. DNAscape does not
launch a browser automatically; set ``DNASCAPE_APP_SHOW=true`` to opt in.
Use the app for inspection and short interactive runs; use Python scripts or
notebooks for reproducible analyses, sweeps, and publication pipelines.

Pages
-----

Overview
   Navigate between Data Manager, Mapping, Simulation, Plotting, Cached Files,
   and Resources.

Data Manager
   Automatically detect uploaded DNAscape profile dependence and convert between
   supported genomic data forms. Missing values can be rejected, preserved,
   interpolated, or filled with zero. Completed conversions provide a direct
   download and remain available in Cached Files for the browser tab.

Mapping
   Configure ``rmap`` inputs, regions, routes, and preview settings, then run
   the resolved route in a cancellable background worker.

Simulation
   Run bounded constant or uploaded firing-rate and fork-speed fields through ``rsim``.
   Progress appears in Preview while the scientific worker runs. Enable
   **Track detailed outputs** only when time courses and space-time fields are
   needed; the app validates memory and workload limits before allocation.

   Plot previews can be enlarged and exported as PNG, SVG, or PDF. Selected
   raw arrays can be downloaded as CSV, TSV, TXT, NPY, or NPZ. The application
   Settings menu controls plot DPI and raw compression; GZIP is the default.

Plotting
   Upload one or more profiles and create profile, heatmap, histogram, or
   box-plot previews with the same plotting controls used by Mapping and
   Simulation.

Cached Files
   Search, sort, download, reopen, or remove uploads, prepared datasets,
   generated files, and completed workflows retained in the current browser
   tab. The oldest entries are removed automatically above 30 items per
   category, 80 items overall, or 256 MB retained memory. The cache is cleared
   when the app process ends.

Resources
   Open the repository, documentation, issue tracker, and citation guidance.

.. figure:: _static/img/app_overview_placeholder.svg
   :width: 760
   :alt: Placeholder for the NiceGUI overview page

   DNAscape application layout.

Constant Simulation Example
---------------------------

1. Open **Simulation**.
2. Keep both inputs set to **Constant**.
3. Choose the genomic interval, resolution, replicate count, and seed.
4. Leave **Track detailed outputs** off for the quickest timing preview, or
   enable it for replicated fraction, fork statistics, initiation/coalescence
   fields, and fork-density heatmaps.
5. Select **Simulate** and monitor progress in Preview.
6. Choose the output to display, then use the hover toolbar or Raw data section
   to export it.

Safety Limits
-------------

The local interface validates numeric inputs, spatial and temporal grid sizes,
replicate-scaled work, and detailed-output memory before calling ``rsim``.
Oversized requests are rejected with guidance to coarsen the resolution,
shorten the interval, or reduce the ensemble.
