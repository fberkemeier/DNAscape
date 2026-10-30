.. _python-notebooks:

Notebook examples
=================


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
