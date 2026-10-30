.. image:: _static/dnascape_logo.svg
   :align: center
   :width: 320px
   :alt: DNAscape

Overview
========

DNAscape is software for preparing genomic data, mapping DNA replication
kinetics and simulating replication programmes. Infer origin firing or fork
speed from compatible profiles, predict replication features, and explore
stochastic S phases and their variability.

Start with the app
------------------

Follow :doc:`installation`, then the :doc:`quickstart` to prepare Repli-seq
data, map a spatial initiation rate and simulate the fitted programme.
The :doc:`appguide` follows the pages and controls of the interface.

Continue in Python
------------------

:doc:`python` explains how to export an app workflow or use the three Python
entry points: ``rdata`` prepares data, ``rmap`` maps replication features and
``rsim`` simulates replication. :doc:`notebooks` provides worked examples.
:doc:`foundations` explains the quantities, units and mathematical assumptions.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   Overview <self>
   gettingstarted
   appguide
   python
   foundations

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
