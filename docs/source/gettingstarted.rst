Getting started
===============


Install and launch the app, then follow one complete workflow from a public
Repli-seq track to an inferred initiation rate and a simulated programme.

.. _installation:

Installation
------------

Run the DNAscape app locally using the same steps as the repository README.

Requirements
~~~~~~~~~~~~

* Python 3.10 or newer (Python 3.13 recommended).
* Git.
* A modern web browser.

1. Download DNAscape
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/fberkemeier/DNAscape.git
   cd DNAscape

Run the remaining commands from this project folder.

2. Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install -r app/requirements.txt

This also installs the DNAscape Python package and its numerical dependencies.

4. Start DNAscape
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m app

Open the address printed in the terminal. Keep that terminal running while
using the app; press ``Ctrl+C`` there to stop it.

For later launches, activate ``.venv`` again and run ``python -m app``.

.. _app-open-browser:

Open the browser automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Windows PowerShell:

.. code-block:: powershell

   $env:DNASCAPE_APP_SHOW = "true"
   python -m app

On macOS and Linux:

.. code-block:: bash

   DNASCAPE_APP_SHOW=true python -m app

Continue with :ref:`quickstart`, or use :doc:`python` for code-based workflows.

.. _quickstart:

Quickstart
----------

This first run uses the app to prepare Repli-seq data, infer a spatial
initiation rate and simulate replication. It connects three app pages:
**Data Manager -> Mapping -> Simulation**.

Start the app using :ref:`installation`. The example uses K562 chromosome 1
and a constant fork speed of 1.4 kb/min.

1. Prepare a timing profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open **Data Manager**.

1. Select **Link**, paste the public K562 Repli-seq BigWig URL below and click
   **Inspect**. A successful inspection replaces the button text with a check.
2. In **Configure**, leave **Apply configuration** on. Set **Prepare as** to
   **Replication timing**, choose **chr1**, use the full chromosome, and set
   **Bin size (bp)** to **1000** and **Aggregation** to **Mean**.
3. Under **Values**, choose **Missing values -> Fill with a value** and use
   **30** as the fill value.
4. Open **Data transforms** and enable **Apply transform**. Leave the formula
   as ``source``. Choose **Value scaling -> Target interval**, with **Minimum**
   **30** and **Maximum** **500**, and enable **Invert**.
5. Click **Convert**. The new result appears at the top of **Generated Files**.
   Inspect it and download the timing profile from **Raw data** as TSV.

.. code-block:: text

   https://hgdownload.gi.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig

The wave signal is relative. Assigning it a 30-500 minute interval is an
illustrative calibration: larger Repli-seq scores become earlier times after
inversion. Use an experimentally justified time scale for quantitative work.
For a Python workflow using bundled data without downloads, see :ref:`pythonmodules`.

2. Infer initiation rate
~~~~~~~~~~~~~~~~~~~~~~~~

Open **Mapping**.

1. Choose **Mapped output -> Initiation rate**, **Dependence -> Space**, and
   **Input factors -> Replication timing + fork speed**.
2. Under **Inputs**, upload the prepared timing profile with **Space**
   dependence. Set fork speed to **Constant**, with **Value** **1.4 kb/min**.
3. In **Region**, turn **Whole region** off. Set **Start** to **40000 kb**,
   **End** to **50000 kb**, and **Resolution** to **1 kb**. Leave **Periodic** off.
4. Click **Map**. Compare the observed timing with the fitted prediction and
   inspect the MSE. Download the initiation-rate profile from **Raw data**.

You can also use a result's **Open in** button to transfer it directly. For
Mapping, choose the target, route and receiving input when prompted.

3. Simulate the fitted programme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open **Simulation**.

1. Upload the mapped initiation rate as a **Profile** with **Space** dependence.
   Keep fork speed **Constant** at **1.4 kb/min**.
2. Enable **Whole region** and use **Resolution** **1 kb/bin**. This uses the
   full uploaded profile, which now represents the mapped 10,000 kb interval.
3. Set **Simulations (replicates)** to **100** and enable **Fix seed** for a
   reproducible run. Enable **Track time courses** to include progression,
   origin firing over time and replicon lengths.
4. Click **Simulate**. In **Preview**, select an output to inspect and download
   the data you need. Compare the ensemble timing with the mapping prediction.

Change one assumption, such as fork speed, and rerun to compare programmes.
The :ref:`simulationengine` guide explains optional fork mechanisms and outputs.

Continue
~~~~~~~~

* :doc:`appguide` groups the page guides and shared controls.
* :ref:`niceguiapp` explains navigation, plots, transfers and settings.
* :ref:`exportworkflows` explains how to reproduce the current app settings in Python.
* :doc:`foundations` explains the equations and assumptions.
* :doc:`python` introduces the direct functions and notebook examples.
