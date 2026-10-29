Installation
============

An online version is a work in progress. To run the DNAscape app locally,
follow the steps below, as in the repository README.

Requirements
------------

* Python 3.10 or newer (Python 3.13 recommended).
* Git.
* A modern web browser.

1. Download DNAscape
--------------------

.. code-block:: bash

   git clone https://github.com/fberkemeier/DNAscape.git
   cd DNAscape

Run the remaining commands from this project folder.

2. Create a virtual environment
-------------------------------

Windows PowerShell:

.. code-block:: powershell

   py -m venv .venv
   .\.venv\Scripts\Activate.ps1

macOS and Linux:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

.. note::

   For BigWig files and links, the app installation selects ``pybigtools`` on
   Python 3.10-3.13 or ``figwig`` on Python 3.14 and newer. Both provide the
   same region and binning controls. Figwig compiles and caches its decoding
   routines on the first read, which can take a few seconds longer.

3. Install the app
------------------

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install -r app/requirements.txt

This also installs the DNAscape Python package and its numerical dependencies.

4. Start DNAscape
-----------------

.. code-block:: bash

   python -m app

Open the address printed in the terminal. Keep that terminal running while
using the app; press ``Ctrl+C`` there to stop it.

To open your default browser automatically, see :ref:`app-open-browser`.
Continue with :doc:`quickstart`, or use :doc:`python` for code-based workflows.
