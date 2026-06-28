Notebooks
=========

Run from the repository root:

.. code-block:: bash

   python -m jupyterlab examples

Sequence
--------

.. list-table::
   :header-rows: 1
   :widths: 26 28 23 23

   * - Notebook
     - Purpose
     - Inputs
     - Outputs
   * - ``examples/0-quickstart.ipynb``
     - First ``rmap`` and ``rsim`` workflow.
     - Bundled timing data.
     - Firing rate, timing, simulation plots.
   * - ``examples/1-data_management.ipynb``
     - Data loading and export.
     - Tables, arrays, bundled data.
     - Prepared arrays and files.
   * - ``examples/2-mapping.ipynb``
     - ``rmap`` route atlas.
     - Timing, firing, speed, RFD.
     - Mapped features and comparisons.
   * - ``examples/3-simulation.ipynb``
     - ``rsim`` controls and outputs.
     - Firing and fork speed.
     - Timing, efficiency, fork counts, heatmaps.
   * - ``examples/4-neat_examples.ipynb``
     - Small focused examples.
     - Example profiles.
     - Compact demonstrations.
   * - ``examples/tests.ipynb``
     - Smoke checks.
     - Bundled data.
     - Minimal pipeline checks.

Related Docs
------------

* :doc:`quickstart` for the first workflow.
* :doc:`datamanagement` for data prep.
* :doc:`mappingfunctions` for ``rmap``.
* :doc:`simulationengine` for ``rsim``.
* :doc:`visualisation` for plots.
* :doc:`api_reference` for signatures.

Practices
---------

* Keep raw inputs unchanged.
* Save prepared data separately.
* Record region, resolution, smoothing, and normalization.
* Set ``seed`` for simulation comparisons.
* Move repeated notebook code into scripts.

Figures
-------

Current docs use placeholders. Good replacements: app screenshots, mapped
firing profiles, RFD comparisons, heatmaps, and observed-versus-predicted
plots from the notebooks.
