Release Notes
=============

Unreleased
----------

* Removed the Directional fork densities route for Mapping's Replicated fraction target.
* Accelerated raw Mapping CSV/TSV downloads with bounded numeric formatting and
  reused immutable result arrays, retaining full precision and existing file contents.
  Large replicated-fraction downloads use faster lossless gzip/ZIP compression.

0.1.1
-----

* Show Reverse order in plot settings only when multiple curves share a plot.

* Reduced first-click delays on Render by preparing shared workflow imports at
  startup when memory headroom permits. Visitor pages remain lazy and isolated;
  low-memory services retain deferred imports.
* Isolated NiceGUI preferences by client and persisted them in namespaced
  browser localStorage. Theme, interface and export changes no longer affect
  other visitors; reset clears only this browser's DNAscape preferences.
* Background calculations and exports receive explicit settings snapshots.
  Headless recipes use independent scopes instead of temporarily changing
  process-wide preview settings. Hosted memory safeguards remain in place.
* Time-only initiation and fork-speed uploads now retain their complete row
  of time samples, without a value-column selector or per-column batch expansion.
* Reduced source interpolation, travel-coordinate and Weibull-field overhead
  when fitting space-time fork speed from space-time initiation and timing.
* Timing uploads no longer select every numeric column for independent fits
  automatically. Space-time initiation uploads retain their complete matrix.
* Accelerated temporal and space-time fork-speed fitting with shared source-cell
  edges, compact characteristic caches and smaller temporal adjoints. Unused
  final adjoints are skipped while preserving fit grids and stopping rules.
* Local replicated-fraction mapping now processes chromosome-scale native grids
  in bounded blocks, with temporary disk-backed outputs and full-resolution exports.
* Reduced travel-coordinate inversion and interpolation-adjoint overhead while
  retaining the numerical model, fit grids, objectives and stopping rules.
* Removed the persistent Render input-limit notices from Mapping and Simulation.
* Improved adaptive-fitting ETA estimates using measured phase costs and the
  remaining work on each refinement grid.
* Corrected periodic directionality smoothing, circular preview extent and
  exact opposing-fork arrival ties in Simulation. Mapping event targets now
  offer only the Initiation rate + Fork speed route.
* Replicated-fraction previews default to 1.5 times the latest local median
  replication time while retaining the complete computed data.
* Removed the prototype badge from the app header.
* Limited Render cached history to 4 MiB per tab and 8 MiB shared per process,
  with configurable retention, workflow history off by default and expired
  session cleanup. Oversized cache entries are skipped before insertion.
  Deleted upload controls also release callbacks that retain page data.
* Render Mapping and Simulation now accept one constant or profile per input,
  with replacement uploads and server checks before batch materialisation.
  Local multiple-constant and multiple-profile workflows remain available.
* Fixed memory pressure during final space-time initiation refinement with
  overlapping spatial FFT neighbourhoods, fewer temporary arrays and adaptive
  output batches. The final accepted proposal skips its unused adjoint.
* Heat-map previews no longer scan the full target for hidden histogram
  controls; refinement progress identifies work at the selected resolution.
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
