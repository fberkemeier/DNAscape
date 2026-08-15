NiceGUI App
===========

Run the local interface from the repository root:

.. code-block:: bash

   python -m pip install -r app/requirements.txt
   python -m app

Open ``http://127.0.0.1:8080`` if the browser does not open automatically.
Use the app for inspection and short interactive runs; use Python scripts or
notebooks for reproducible analyses, sweeps, and publication pipelines.

Pages
-----

Overview
   Navigate between Data Manager, Mapping, Simulation, Plotting, Cached Files,
   and Resources.

Data Manager
   Explore file conversion and preprocessing controls. Backend conversion is
   being connected incrementally.

Mapping
   Configure ``rmap`` inputs, regions, routes, and preview settings. Backend
   execution is being connected incrementally.

Simulation
   Run bounded constant firing-rate and fork-speed ensembles through ``rsim``.
   Progress appears in Preview while the scientific worker runs. Enable
   **Track detailed outputs** only when time courses and space-time fields are
   needed; the app validates memory and workload limits before allocation.

   Plot previews can be enlarged and exported as PNG, SVG, or PDF. Selected
   raw arrays can be downloaded as CSV, TSV, TXT, NPY, or NPZ with optional
   gzip compression.

Plotting
   Explore controls for profiles, heatmaps, and histograms. Backend plotting
   is being connected incrementally.

Cached Files
   Inspect the intended generated-file and retained-session layout while cache
   persistence is being connected.

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

