Installation
============

Run the DNAscape app locally using the same steps as the repository README.

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

Choose an installed Python version (3.13 is recommended).

Windows PowerShell:

.. code-block:: powershell

   $pythonVersion = "3.13"
   py "-$pythonVersion" -m venv .venv
   .\.venv\Scripts\Activate.ps1

macOS and Linux:

.. code-block:: bash

   python3.13 -m venv .venv
   source .venv/bin/activate

When changing Python versions or computers, create a fresh environment and
reinstall the dependencies. Do not reuse or copy an existing ``.venv``.

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

For later launches, activate ``.venv`` again and run ``python -m app``.

.. _app-open-browser:

Open the browser automatically
------------------------------

In Windows PowerShell:

.. code-block:: powershell

   $env:DNASCAPE_APP_SHOW = "true"
   python -m app

On macOS and Linux:

.. code-block:: bash

   DNASCAPE_APP_SHOW=true python -m app

Continue with :doc:`quickstart`, or use :doc:`python` for code-based workflows.
