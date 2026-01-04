Quickstart
==========

Here we show a minimal DNAscape workflow: load a replication timing profile,
infer a firing-rate profile, and run a stochastic simulation to generate
derived quantities such as replication timing, origin efficiency, inter-origin distances, and the
replicated fraction.

Mapping replication timing to firing rate
-----------------------------------------

Replication timing profiles provide a natural entry point for kinetic
inference. In this example, a replication timing profile for chromosome 1 is
loaded and visualised.

.. code-block:: python

   timedata = loadcsv("example", example="replication_timing_hESC", chr_number=1)

which can be plotted

.. code-block:: python

   plotf(timedata, xtitle="Chromosome position (kb)", ytitle="Replication time (min)", invyQ=True)

.. figure:: _static/img/plot_quick1.png
   :width: 700
   :alt: Replication timing profile example

From the replication timing profile, a firing-rate profile is inferred using
the DNAscape kinetic mapping routines.

.. code-block:: python

   frates = rfit("replication_timing", "firing_rate", source=timedata)

.. code-block:: python

   plotf(frates, xtitle="Chromosome position (kb)", ytitle="Firing rate (min⁻¹)")

.. figure:: _static/img/plot_quick2.png
   :width: 700
   :alt: Inferred firing-rate profile example

The inferred firing-rate profile can be mapped back to a replication timing
profile to check self-consistency of the kinetic mapping.

.. code-block:: python

   timepred = rfit("firing_rate", "replication_timing", source=frates)

A direct comparison between the observed timing and the timing predicted from
the inferred firing-rate profile can be made on a subregion of the chromosome.

.. code-block:: python

   plotf(
       timedata[10000:40000],
       timepred[10000:40000],
       labels=["Timing data", "Timing predicted"],
       xtitle="Chromosome position (kb)",
       ytitle="Time (min)",
       invyQ=True,
   )

.. figure:: _static/img/plot_quick3.png
   :width: 700
   :alt: Timing data compared with timing predicted from inferred firing rate

Simulation from an inferred firing-rate profile
-----------------------------------------------

Given an inferred firing-rate profile, DNAscape can simulate replication
dynamics across an ensemble of stochastic realisations.

.. code-block:: python

   simres = rsim(frates, time_statsQ=True, sim_number=100)

These simulations produce genome-wide observables such as replication timing, inter-origin
distances, fork counts, and replicated fraction. These can be extracted from the simulation as follows

.. code-block:: python

   timesim = simres["replication_timing"]
   iod     = simres["inter_ori_distances"]
   frep    = simres["fraction_replicated"]
   forka   = simres["active_forks"]
   nroris  = simres["num_oris"]
   effic   = simres["efficiency"]

A three-way comparison can be made between the observed timing, the timing
predicted from the inferred firing-rate profile, and the timing emerging from
stochastic simulations.

.. code-block:: python

   plotf(
       timedata[10000:40000],
       timepred[10000:40000],
       timesim[10000:40000],
       labels=["Timing data", "Timing predicted", "Timing simulated"],
       xtitle="Chromosome position (kb)",
       ytitle="Time (min)",
       invyQ=True,
   )

.. figure:: _static/img/plot_quick4.png
   :width: 700
   :alt: Timing data, predicted timing, and simulated timing comparison

The distribution of inter-origin distances provides a compact summary of
origin spacing emerging from the inferred kinetics.

.. code-block:: python

   ploth(iod, xtitle="Inter-origin distance (kb)", bin_size=10, xmax=1000)

.. figure:: _static/img/plot_quick5.png
   :width: 700
   :alt: Inter-origin distance distribution example

Temporal evolution of replication can be visualised by jointly plotting the
fraction of replicated DNA and the number of active forks as a function of
time.

.. code-block:: python

   plotf(
       frep, forka,
       xtitle="Time (min)",
       labels=["Replicated fraction", "Active forks"],
       dual_axis=True,
   )

.. figure:: _static/img/plot_quick6.png
   :width: 700
   :alt: Replicated fraction and active forks over time example

Origin efficiency can be visualised alongside the inferred firing-rate profile.

.. code-block:: python

   plotf(
       effic, frates,
       xtitle="Chromosome position (kb)",
       labels=["Origin efficiency", "Firing rate"],
       dual_axis=True,
   )

.. figure:: _static/img/plot_quick7.png
   :width: 700
   :alt: Origin efficiency and firing rate (dual axis)

The distribution of the total number of origins firing per simulation can be
summarised using a whisker plot (optionally with a violin overlay).

.. code-block:: python

   plotw(nroris, ytitle="Number of origins", labels=["hESC"], violinQ=True)

.. figure:: _static/img/plot_quick8.png
   :width: 700
   :alt: Number of origins distribution across simulations
