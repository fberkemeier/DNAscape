Python
======

Use the app to establish a workflow, then move to Python for repeatable runs,
parameter sweeps and integration with other analyses. The three entry points
are ``rdata`` for preparation, ``rmap`` for mapping and ``rsim`` for simulation.
The :doc:`installation` app setup already installs the Python package.

For package-only use from the repository root:

.. code-block:: bash

   python -m pip install .

Use a workflow exported by the app
----------------------------------

Enable **Settings -> Interface -> Developer mode** and select **Code** on a
workflow page. The download is one ``.py`` file, containing editable settings
and input/output folder definitions.

Uploads are optional: missing inputs use placeholder filenames. Edit these
paths to use files already stored locally, or place the files in the input folder.

Place the listed original files in its input folder, adjust paths as needed,
and run it with the same DNAscape version. Data Manager scripts reproduce
preparation and transforms; Mapping and Simulation scripts reproduce their
numerical outputs. Analysis code reproduces its selected preparation and
output settings, including requested figure exports.

A complete three-module example
-------------------------------

Run this script from the repository root. It uses bundled hESC chromosome 1
data, writes to a separate output folder and needs no downloads. The selected
1,000 kb region is averaged from 1 kb input into 5 kb bins.

.. code-block:: python

   from pathlib import Path

   import numpy as np
   from dnascape import rdata, rload, rmap, rsim, rsave, select_profile_region

   INPUT_DIRECTORY = Path("examples/data/example")
   OUTPUT_DIRECTORY = Path("outputs/dnascape_example")
   OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

   # rdata: select a chromosome and convert its ready-formatted values.
   prepared_path = OUTPUT_DIRECTORY / "chr1_timing.npz"
   rdata(
       INPUT_DIRECTORY / "replication_timing_hESC.csv",
       target="npz",
       output_path=prepared_path,
       chromosome="chr1",
       data_form="space",
   )
   full_timing = np.asarray(rload(prepared_path, data_form="space")).reshape(-1)
   dx_kb = 5.0
   timing, positions_kb = select_profile_region(
       full_timing,
       raw_resolution_kb=1.0,
       interval_kb=(40_000.0, 41_000.0),
       output_resolution_kb=dx_kb,
       resample_method="mean",
   )

   # rmap: the direct Python interface uses initiation density per kb/min.
   fit = rmap(
       source={"replication_timing": timing, "fork_speed": 1.4},
       target="initiation_rate",
       assumptions={
           "target_form": "space",
           "resolution_x": dx_kb,
           "maxiter": 10,
           "verbose": False,
       },
       return_info=True,
   )
   ir_density = np.asarray(fit["values"])

   # rsim: convert density to potential initiation rate per spatial bin/min.
   simulation = rsim(
       source={"initiation_rate": ir_density * dx_kb, "fork_speed": 1.4},
       assumptions={
           "resolution_x": dx_kb,
           "sim_number": 32,
           "seed": 17,
           "time_statsQ": True,
           "time_grid": np.arange(0.0, 1201.0, 5.0),
           "max_rep_time": 1200.0,
           "verbose": False,
       },
   )

   rsave(ir_density, OUTPUT_DIRECTORY / "initiation_density_x.tsv", data_form="space")
   np.savez_compressed(
       OUTPUT_DIRECTORY / "simulation_results.npz",
       positions_kb=positions_kb,
       resolution_kb=dx_kb,
       seed=17,
       initiation_density_per_kb_min=ir_density,
       replication_timing=simulation["replication_timing"],
       time_min=simulation["time_grid"],
       origin_firing=simulation["origin_firing"],
       replicon_lengths_kb=simulation["replicon_lengths"],
   )
   print(f"Wrote results to {OUTPUT_DIRECTORY.resolve()}")

The exported initiation density is labelled explicitly. To upload it to the
app's initiation-rate input at this 5 kb resolution, convert it to a per-bin
rate by multiplying by 5; app-to-app transfers already use that convention.

Inputs and assumptions
----------------------

Use a dictionary for ``source`` and a separate dictionary for ``assumptions``.
Specify ``target_form`` when fitting a constant, spatial, temporal or space-time
quantity. Time profiles use shape ``(1, T)``; space-time fields use ``(n, T)``.
Direct arrays need explicit spatial and temporal resolutions.

``rmap(..., return_info=True)`` returns values and route/fit metadata. ``rsim``
returns a dictionary of requested outputs. To see supported quantities and
controls for your installed version:

.. code-block:: python

   from dnascape import rmap, rsim

   rmap.targets()
   rmap.requirements(target="fork_speed")
   rsim.input()
   rsim.output()

Time-dependent and space-time fits have route-specific controls. Prefer the
settings shown by the selected route or an app-exported script over copying
all settings between different target dependences.

Genomic tracks and shared app workflows
---------------------------------------

``rdata`` and ``rtrack`` can bin bedGraph data using bp coordinates. For the
core Python BigWig importer, install the optional reader separately:

.. code-block:: bash

   python -m pip install ".[genomics]"

That lower-level path uses ``pyBigWig``. The app and exported Data Manager
workflows instead use the version-selected ``pybigtools``/``figwig`` readers
installed by the app requirements. Use an app-exported script to reproduce
that reader choice and its preparation controls.

Shared preparation, region transforms and workflow execution also live in
``dnascape.workflows`` and can be used without starting the interface. The
notebooks demonstrate formulas, masks, input modulation and coordinate-aware
conversion using these helpers.

.. _python-notebooks:

Notebooks
---------

The `examples folder <https://github.com/fberkemeier/DNAscape/tree/main/examples>`_
contains independent, runnable notebooks with explicit paths, axes, units and
seeds. The main examples work offline; public track downloads are opt-in.

.. list-table:: Suggested sequence
   :header-rows: 1
   :widths: 40 60

   * - Notebook
     - Focus
   * - ``0-quickstart.ipynb``
     - Prepare bundled timing, map initiation rate, simulate and export.
   * - ``1-data_management.ipynb``
     - Formats, resolution, genomic tracks, transforms, modulation and masks.
   * - ``2-mapping.ipynb``
     - Kinetic dependences, inverse fits, forward predictions and event densities.
   * - ``3-simulation.ipynb``
     - Ensemble outputs, time courses, space-time fields and optional mechanisms.
   * - ``4-neat_examples.ipynb``
     - Masked perturbations, covariate modulation, parameter sweeps and comparisons.
   * - ``5-plotting.ipynb``
     - Static figures and interactive exports, including circular profiles.
   * - ``tests.ipynb``
     - Small shape, unit, reproducibility, mask and file-round-trip checks.

To open them from the repository root:

.. code-block:: bash

   python -m pip install jupyterlab
   python -m jupyterlab examples

Choose the environment where DNAscape is installed and use **Restart Kernel
and Run All**. Keep prepared inputs, numerical outputs and model settings
together when adapting an example to a new dataset.
