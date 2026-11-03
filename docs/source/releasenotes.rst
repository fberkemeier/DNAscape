Release Notes
=============

Unreleased
----------

* Added Render-only memory detection, allocation checks, shared heavy-job
  admission and bounded caches, with warnings linking to local installation.
* Restored fast convolutions for hosted timing fits, with bounded working
  arrays, reusable kernel caches and derivatives only for accepted proposals.
  Time-dependent fits batch uniform histories; space-time initiation fits
  with constant fork speed use exact cone convolutions in FFT time blocks.
* Corrected hosted BigWig conversion when inspection has no array shape;
  memory admission uses the resolved genomic bins.

* Reorganised the documentation into Getting started, App guide, Python and
  notebooks, and Core concepts, with each chapter's subsections on one page.
* Renamed the shared app controls section to User interface.
* Added the unified mapping framework and the Nature Communications paper
  to Core concepts.
* Updated the main page with a concise contents list, overview, citation and
  contact information.
* Aligned Installation and Quickstart with the README's app installation
  command and current data preparation, mapping and simulation walkthrough.
* Simplified Installation by removing virtual environment and automatic
  browser-opening instructions, and capitalised section and subsection titles.
