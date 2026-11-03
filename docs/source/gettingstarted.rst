Getting Started
===============


Install and launch the app, then follow one complete workflow from a public
Repli-seq track to an inferred initiation rate and a simulated programme.

.. _installation:

Installation
------------

DNAscape is best used through its interactive browser app, built with NiceGUI.
The steps below follow the repository README and run the app locally.

Requirements
~~~~~~~~~~~~

* Python 3.10 or newer (Python 3.13 recommended).
* Git.
* Pip.
* A modern web browser.

1. Download the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/fberkemeier/DNAscape.git
   cd DNAscape

Run the remaining commands from this project folder.

2. Install and run the app
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install ".[app]"
   python -m app

This installs the DNAscape Python package and its app dependencies.
Open the address printed in the terminal. Keep that terminal running while
using the app; press ``Ctrl+C`` there to stop it.

For later launches, run ``python -m app`` from the repository folder.

Continue with :ref:`quickstart`, or use :doc:`python` for code-based workflows.

.. _quickstart:

Quickstart
----------

This first run follows the README example: prepare a replication-timing
profile from public Repli-seq data, infer a spatial initiation rate, and
simulate the fitted replication programme. It connects three app pages:
**Data Manager -> Mapping -> Simulation**.

Start the app using :ref:`installation`. The example uses K562 chromosome 1
and a constant fork speed of 1.4 kb/min.

I. Managing data
~~~~~~~~~~~~~~~~

With the app open, select **Data Manager** in the left-hand menu.

1. Download the `K562 Repli-seq BigWig file`_ from ENCODE and **Upload** it.
   Alternatively, choose **Link**, select
   **Example: ENCODE Repli-seq (bigWig) - K562** to fill in the URL, and click
   **Inspect**. You can also paste the file's URL directly into the Link field.
2. In **Configure**, leave **Apply configuration** on. Set **Prepare as** to
   **Replication timing**, choose **chr1**, use the full chromosome, and set
   **Bin size (bp)** to **1000** and **Aggregation** to **Mean**.
   Under **Values**, set **Missing values** to **Fill with a value** and
   **Fill value** to **30** minutes.
3. Open **Data transforms** and enable **Apply transform**. Leave the formula
   as ``source``. Choose **Value scaling -> Target interval**, with **Minimum**
   **30** and **Maximum** **500**, and enable **Invert**.
4. Click **Convert**. Inspect the plot, then download the timing profile as
   TSV from **Generated Files -> Raw data**.

.. _K562 Repli-seq BigWig file: https://hgdownload.gi.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig

The wave signal is relative. Assigning it a 30-500 minute interval is an
illustrative calibration: larger Repli-seq scores become earlier times after
inversion. Use an experimentally justified time scale for quantitative work.
For a Python workflow using bundled data without downloads, see :ref:`pythonmodules`.

II. Mapping features
~~~~~~~~~~~~~~~~~~~~

Select **Mapping** in the left-hand menu to infer a spatial origin
initiation-rate profile that fits the prepared timing data.

1. Within **Target**, set **Mapped output** to **Initiation rate**,
   **Dependence** to **Space**, and **Input factors** to
   **Replication timing + fork speed**. This selects the route that infers
   initiation rate from supplied timing and fork speed.
2. Under **Inputs**, upload the prepared timing profile (TSV) and confirm its
   **Space** dependence. Set **Fork speed** to **Constant**, with a value of
   **1.4 kb/min**.
3. In **Region**, leave **Whole region** off. Set **Start** to **40000 kb**,
   **End** to **50000 kb**, and **Resolution** to **1 kb**. Leave **Periodic**
   off, then click **Map**. Compare observed and fitted timing using the
   **Output** selector in **Preview**, inspect the MSE, and download the fitted
   initiation-rate profile from **Raw data**.

You can also use a result's **Open in** button to transfer it directly. For
Mapping, choose the target, route and receiving input when prompted.

III. Simulating DNA replication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Move to **Simulation** to simulate replication using the inferred
origin-firing profile.

1. Under **Initiation rate**, choose **Profile** and upload the mapped profile.
   DNAscape automatically detects its **Space** dependence. Keep **Fork speed**
   **Constant** at **1.4 kb/min**.
2. In **Region**, enable **Whole region** and keep **Resolution** at
   **1 kb/bin**. This uses the full uploaded profile, representing the mapped
   10,000 kb interval. In **Simulation settings**, set **Simulations
   (replicates)** to **100**; optionally enable **Fix seed** for reproducible
   runs.
3. Click **Simulate**, explore the outputs in **Preview**, and download the
   data or figures you need. Compare the ensemble timing with the mapping
   prediction, then change the fork speed, replicate count or other settings
   and rerun to compare outcomes.

Optionally enable **Track time courses** in Simulation settings to include
replicated fraction, origin firing over time and replicon lengths.
The :ref:`simulationengine` guide explains optional fork mechanisms and outputs.

Continue
~~~~~~~~

* :doc:`appguide` groups the page guides and shared controls.
* :ref:`niceguiapp` explains navigation, plots, transfers and settings.
* :ref:`exportworkflows` explains how to reproduce the current app settings in Python.
* :doc:`foundations` explains the equations and assumptions.
* :doc:`python` introduces the direct functions and notebook examples.
