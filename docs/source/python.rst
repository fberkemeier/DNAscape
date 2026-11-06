Python And Notebooks
====================


Use the app to establish a workflow, then continue in Python for repeatable
runs, parameter sweeps and integration with other analyses. The app
installation already installs the DNAscape Python package.

Start with :ref:`exportworkflows` to reproduce the current app options.
:ref:`pythonmodules` introduces ``rdata``, ``rmap`` and ``rsim`` directly;
:ref:`notebooks` lists the worked examples.

.. _exportworkflows:

Export A Workflow From The App
------------------------------

Enable **Settings -> Interface -> Developer mode** and select **Code** on a
workflow page: Data Manager, Mapping, Simulation or Analysis. The download is
one ``.py`` file with grouped inputs, settings and outputs, plus editable
input/output folder definitions. It runs with the DNAscape package used to
export it; the web app is not required.

Uploads are optional: missing inputs use placeholder filenames. Edit these
paths to use files already stored locally, or place the files in the input folder.

Original uploads are included losslessly in the script and restored to its
input folder on the first run. Existing local files always take precedence;
edit a source path, column or array key to use different data. Set
``USE_EMBEDDED_INPUTS = False`` to disable upload restoration. Missing upload
placeholders remain empty until you supply a real file. Prepared Mapping
arrays are also included, with a local prepared file taking precedence.

Run the script with the same DNAscape version. Data Manager scripts reproduce
preparation and transforms; Mapping and Simulation scripts reproduce their
numerical outputs. Analysis code reproduces its selected preparation and
output settings, including requested figure exports.


The exported script calls the shared workflow layer, retaining preparation,
resampling, transforms and output choices from the page. A direct ``rmap`` or
``rsim`` call uses the same scientific engine but needs its inputs and
assumptions supplied explicitly; see :ref:`pythonmodules`.

.. code-block:: bash

   python data_manager.py

Use the downloaded filename for Mapping, Simulation or Analysis. The script
defines ``INPUT_DIRECTORY`` and ``OUTPUT_DIRECTORY`` near the top; outputs
are written relative to those folders rather than the terminal's current
directory. Keep the script, source files and DNAscape version with the result.

Scripts also work when pasted into a notebook, using its working directory.
For repeated runs, import the downloaded module, copy its ``RECIPE``
(``MAPPING`` for Mapping), edit the copy, and call
``main(recipe=your_recipe, input_directory=..., output_directory=...)``.
An integer simulation seed reproduces the random sequence; ``None`` starts
independent simulations. Code downloads capture settings without running the
calculation or changing the current results.

.. _pythonmodules:

Use ``rdata``, ``rmap`` And ``rsim``
------------------------------------


The three public entry points are ``rdata`` for data preparation, ``rmap`` for
mapping replication features and ``rsim`` for stochastic simulation. Use
explicit inputs and assumptions, and keep units and coordinates with each
array. :ref:`concept-grids` and :ref:`concept-units` define their conventions.

For package-only use from the repository root:

.. code-block:: bash

   python -m pip install .

A complete three-module example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

The exported initiation density is labelled explicitly. When moving data
between Python and the app, check both the rate units and coordinate/bin-width
metadata. App-to-app transfers handle their own rate conversions; see
:ref:`concept-units`.

Inputs and assumptions
~~~~~~~~~~~~~~~~~~~~~~

Use a dictionary for ``source`` and a separate dictionary for ``assumptions``.
Specify ``target_form`` when fitting a constant, spatial, temporal or space-time
quantity. Time profiles use shape ``(1, T)``; space-time fields use ``(n, T)``.
Direct arrays need explicit spatial and temporal resolutions.

Spatial stalling and recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rsim`` accepts ``stall_rate_per_kb`` as a scalar or spatial vector in stalls/kb.
Supply recovery using either ``tau`` (positive mean time in minutes, including
infinity) or ``stall_recovery_rate`` (non-negative inverse minutes). Zero rate
means a permanent stall; omitted recovery also means no restart. Supplying both
representations raises an error. Direct arguments and legacy scalar assumptions
continue to work; spatial movement profiles belong in ``source``.

.. code-block:: python

   import numpy as np
   from dnascape import rsim

   x = np.arange(256)
   density = np.where((x >= 64) & (x < 96), 0.3, 0.01)
   recovery = 0.1 * (1 + 0.4 * np.sin(x / 20))
   result = rsim(
       source={
           "initiation_rate": 0.01,
           "fork_speed": 1.4,
           "stall_rate_per_kb": density,
           "stall_recovery_rate": recovery,
       },
       assumptions={"sim_number": 32, "resolution_x": 1.0, "seed": 7,
                    "stall_shared_directions": True, "fork_speed_std": 0.2},
   )

For each bin, stalls are sampled with probability ``1 - exp(-s(x) * resolution_x)``.
Encountered pauses are exponential with local mean ``tau(x) = 1 / k_r(x)`` and
are added to the prescribed travel time. Shared-direction mode reuses both the
sites and sampled pauses; stochastic speed sampling remains independent.

The workflow request ``ConstantSimulationRequest`` accepts spatial
``stall_rate_per_kb``, ``stall_duration`` (the workflow spelling of ``tau``), or
``stall_recovery_rate``. Native uploaded profiles use ``source_resolution_x``
and the same region/resampling rules as initiation and speed. The application
and Developer scripts use this workflow. Recovery-time uploads accept positive
infinity for permanent stalls; recovery-rate profiles represent these with zero.

``max_rep_time`` censors arrivals after the cutoff as infinity. Fraction outputs
therefore remain below one for incomplete realizations. ``completed`` and
``incomplete_simulations`` distinguish complete realizations; ``completion_time``
and ``s_phase_duration`` are infinity when incomplete and NaN if no origin fired.
The existing finite-arrival timing mean/std definitions are retained, with
``replication_timing_observed_fraction`` reporting the fraction observed per bin.
See ``examples/spatial_stalling.py`` for a runnable example with permanent barriers.

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

Genomic tracks
~~~~~~~~~~~~~~

``rdata`` and ``rtrack`` can bin bedGraph data using bp coordinates. For the
core Python BigWig importer, install the optional reader separately:

.. code-block:: bash

   python -m pip install ".[genomics]"

That lower-level path uses ``pyBigWig``. The app and exported Data Manager
workflows instead use the version-selected ``pybigtools``/``figwig`` readers
installed by the app requirements. Use an app-exported script to reproduce
that reader choice and its preparation controls.

.. _notebooks:

.. _python-notebooks:

Notebook Examples
-----------------


The `examples folder <https://github.com/fberkemeier/DNAscape/tree/main/examples>`_
contains the following notebooks. Use the same environment as your DNAscape
installation. The examples currently use data paths from the repository checkout.

.. list-table:: Suggested sequence
   :header-rows: 1
   :widths: 40 60

   * - Notebook
     - Focus
   * - `1-quickstart.ipynb <https://github.com/fberkemeier/DNAscape/blob/main/examples/1-quickstart.ipynb>`_
     - Prepare timing, infer initiation rate and simulate replication.
   * - `2-mapping.ipynb <https://github.com/fberkemeier/DNAscape/blob/main/examples/2-mapping.ipynb>`_
     - Mapping routes, predictions and fitted replication features.
   * - `3-simulation.ipynb <https://github.com/fberkemeier/DNAscape/blob/main/examples/3-simulation.ipynb>`_
     - Individual S phases, ensemble outputs and optional mechanisms.
   * - `4-data_management.ipynb <https://github.com/fberkemeier/DNAscape/blob/main/examples/4-data_management.ipynb>`_
     - File formats, genomic tracks, binning and value processing.

To open JupyterLab from the repository root:

.. code-block:: bash

   python -m pip install jupyterlab
   python -m jupyterlab examples
