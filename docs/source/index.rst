.. image:: _static/dnascape_logo.svg
   :align: center
   :width: 320px
   :alt: DNAscape

DNAscape
========

DNAscape is software for preparing genomic data, mapping DNA replication
kinetics and simulating replication programmes. Start with experimental
profiles or a model, infer the features you need, and compare the predicted
behaviour with stochastic S phases.

Three modules, one workflow
---------------------------

**rdata -- prepare data**
   Convert files and genomic tracks into profiles with a defined region,
   resolution and meaning. In the app, use :doc:`datamanagement`.

**rmap -- map replication features**
   Infer initiation rate or fork speed, or predict timing, directionality and
   other observables under explicit assumptions. In the app, use
   :doc:`mappingfunctions`.

**rsim -- simulate replication**
   Generate stochastic replication programmes and quantify their mean behaviour
   and variability. In the app, use :doc:`simulationengine`.

The app brings these workflows together without writing code. The same
modules are available from :doc:`python` for scripts, notebooks and parameter
sweeps. :doc:`foundations` explains their biological and mathematical basis.

Start here
----------

Follow :doc:`installation`, then the app-based :doc:`quickstart`. The page
names and controls in these guides match the DNAscape interface.

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   niceguiapp
   datamanagement
   mappingfunctions
   simulationengine
   foundations
   python

Citation and contact
--------------------

If you use DNAscape in research, cite:

   Francisco Berkemeier and Michael A. Boemo (2026).
   *DNAscape: Software for mapping and simulating DNA replication kinetics.*
   bioRxiv.

Include the software version and model settings used. Questions, bugs and
feature requests can be submitted through
`GitHub Issues <https://github.com/fberkemeier/DNAscape/issues>`_ or sent to
`Francisco Berkemeier <mailto:fp409@cam.ac.uk>`_. DNAscape is available under
the `MIT License <https://github.com/fberkemeier/DNAscape/blob/main/LICENSE>`_.
