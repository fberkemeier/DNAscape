Installation
============

DNAscape requires Python 3.9 or later.

Package
-------

Released package:

.. code-block:: bash

   pip install dnascape

Development install:

.. code-block:: bash

   git clone https://github.com/fberkemeier/DNAscape.git
   cd DNAscape
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .

Windows PowerShell:

.. code-block:: powershell

   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -U pip
   python -m pip install -e .

Notebooks
---------

.. code-block:: bash

   pip install notebook jupyterlab ipykernel
   python -m ipykernel install --user --name dnascape --display-name "DNAscape"
   python -m jupyterlab examples

Start with ``examples/0-quickstart.ipynb``.

NiceGUI Interface Prototype
---------------------------

.. code-block:: bash

   pip install -r app/requirements.txt
   python -m app

The interface includes Overview, Data Manager, Mapping, Simulation, Plotting,
Cached Files, and Resources. Constant-input Simulation runs are connected to
``rsim`` with bounded optional detailed outputs and plot/raw-data downloads.

Docs
----

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html

Windows:

.. code-block:: powershell

   python -m pip install -r docs\requirements.txt
   python -m sphinx -b html docs\source docs\build\html

Open ``docs/build/html/index.html``.

Tracks
------

BigWig import needs ``pyBigWig``:

.. code-block:: bash

   pip install pyBigWig

bedGraph import uses pandas only.

Verify
------

.. code-block:: python

   import dnascape

   print(dnascape.rmap.targets().head())
   print(dnascape.rsim.output().head())

CLI:

.. code-block:: bash

   dnascape --help
