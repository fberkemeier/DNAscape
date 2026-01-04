.. DNAscape documentation master file, created by
   sphinx-quickstart on Thu Dec 18 00:26:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DNAscape
========

Software for simulating and mapping DNA replication kinetics.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   quickstart
   foundations
   simulationengine
   mappingfunctions
   datamanagement
   visualisation
   examples
   additionalresources
   releasenotes

Overview
-----------------------------------------

DNAscape is a computational framework for modelling and analysing genome-wide DNA replication kinetics. It provides a unified toolkit to map between experimentally accessible replication observables such as replication timing (e.g., Repli-seq), origin firing properties (Ini-seq, SNS-seq), fork dynamics, and replicated fraction, using mechanistically grounded kinetic models inspired by the Kolmogorov-Avrami-Mehl-Johnson framework.

At its core, DNAscape implements bidirectional mappings between replication observables, enabling inference of origin firing rates from replication timing data, forward prediction of replication timing from inferred kinetics, and stochastic simulation of replication dynamics across entire chromosomes or genomes. These simulations yield emergent quantities including origin efficiency, inter-origin distances, fork counts, temporal replication profiles, among others.

DNAscape is designed to be modular and extensible. It supports chromosome-resolved analyses, various resolutions, ensemble simulations, and flexible visualisation, and is intended to integrate naturally with modern data-driven approaches. The framework aims to bridge experimental data and quantitative theory, providing interpretable, scalable tools for studying DNA replication under both physiological and perturbed conditions.

Publications
-----------------------------------------

If you use DNAscape for your research, please cite the publication appropriate for the version you used:

Bugs, Questions, and Comments
-----------------------------------------

If you encounter any issues or have questions about how to use the software, please open a `GitHub issue <https://github.com/fberkemeier/DNAscape/issues>`_. Feedback is a valuable part of the ongoing development. For comments or suggestions regarding improvements to the software or the documentation, you may also contact Francisco Berkemeier by email at `fp409@cam.ac.uk <mailto:fp409@cam.ac.uk>`_.