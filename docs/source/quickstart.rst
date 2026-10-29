Quickstart
==========

This first run uses the app to prepare Repli-seq data, infer a spatial
initiation rate and simulate replication. It connects the three modules:
**Data Manager (rdata) -> Mapping (rmap) -> Simulation (rsim)**.

Start the app using :doc:`installation`. The example uses K562 chromosome 1
and a constant fork speed of 1.4 kb/min.

1. Prepare a timing profile
---------------------------

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
For a workflow using bundled data without downloads, see :ref:`python-notebooks`.

2. Infer initiation rate
------------------------

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
--------------------------------

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
The :doc:`simulationengine` guide explains optional fork mechanisms and outputs.

Continue
--------

* :doc:`niceguiapp` explains navigation, transfers, settings and code export.
* :doc:`datamanagement`, :doc:`mappingfunctions` and :doc:`simulationengine`
  describe each workflow in more detail.
* :doc:`foundations` explains the equations and assumptions.
* :doc:`python` provides a runnable script and the notebook sequence.
