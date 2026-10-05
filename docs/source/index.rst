.. image:: _static/dnascape_logo.svg
   :align: center
   :width: 50%
   :alt: DNAscape logo

DNAscape
========

DNAscape maps and simulates DNA replication kinetics from genomic data. It
turns timing, initiation, fork-direction, and custom tracks into kinetic
features such as firing rate, fork speed, replicated fraction, fork density,
origin efficiency, and completion time.

Use it as a Python package for reproducible analysis, or as a NiceGUI app
for interactive data conversion, mapping, simulation, plotting, and export.

Quick Links
-----------

* :doc:`quickstart`: minimal package and app workflow.
* :doc:`mappingfunctions`: infer and translate features with ``rmap``.
* :doc:`simulationengine`: simulate replication programmes with ``rsim``.
* :doc:`datamanagement`: prepare tables, arrays, BigWig, and bedGraph tracks.
* :doc:`niceguiapp`: use the interactive app.
* :doc:`api_reference`: public functions and feature names.

rmap / rsim
-----------

``rmap`` (:doc:`mappingfunctions`)
   Maps between replication features under explicit assumptions. Typical use:
   infer firing rate from replication timing and fork speed, or predict timing
   from kinetic inputs.

``rsim`` (:doc:`simulationengine`)
   Simulates stochastic S phases from firing rate and fork speed. Typical use:
   compare perturbations, estimate origin efficiency, and generate timing,
   fork, and replicated-fraction outputs.

Conventions
-----------

DNAscape uses regular arrays:

* space profiles: ``(n,)``;
* time profiles: ``(1, T)``;
* space-time matrices: ``(n, T)``.

``resolution_x`` is genomic bin size in kb. ``resolution_t`` is time step in
minutes. Use ``seed`` for reproducible simulations.

Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   foundations
   datamanagement
   mappingfunctions
   simulationengine
   niceguiapp
   notebooks
   examples
   visualisation
   api_reference
   troubleshooting
   releasenotes

Citation
--------

If you use DNAscape in research, cite the relevant publication or software
release and include the version or commit hash used. For questions or issues,
use the `GitHub issue tracker <https://github.com/fberkemeier/DNAscape/issues>`_
or contact Francisco Berkemeier at `fp409@cam.ac.uk <mailto:fp409@cam.ac.uk>`_.
