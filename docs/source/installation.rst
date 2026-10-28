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

The interface requires Python 3.10 or newer. BigWig support is installed
automatically: ``pybigtools`` on Python 3.10–3.13, or ``figwig`` on Python 3.14
and newer. Both readers support the same region, binning and aggregation controls
for uploads and public links.

.. code-block:: bash

   pip install -r app/requirements.txt
   python -m app

The interface includes Overview, Data Manager, Mapping, Simulation, Plotting,
Cached Files, and Resources. Simulation runs are connected to ``rsim`` with
separate optional time-course and space-time outputs, bounded retained previews,
and plot/raw-data downloads.

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

The app's BigWig reader is included in ``app/requirements.txt``. For direct use
of the lower-level ``dnascape.io`` BigWig functions, install ``pyBigWig``:

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
