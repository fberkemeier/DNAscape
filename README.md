<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is a Python framework for modelling how DNA replication unfolds across chromosomes, genomic regions, and circular domains such as ecDNA.

The central idea is biological and simple: replication timing is not just a signal to plot. It is the visible outcome of where origins fire, when they fire, how forks move, where forks meet, and how these processes vary across cells or molecules. DNAscape provides tools to connect those layers under explicit assumptions.

It is designed for two closely related tasks:

- **Mapping**: translate between experimentally accessible replication features, such as replication timing, origin activity, fork directionality, and other latent kinetic quantities.
- **Simulation**: generate stochastic replication programs from defined assumptions about origin firing, fork progression, fork speed, circularity, and replication stress.

Together, these make it possible to ask not only "what does the data look like?", but "what replication mechanism could have produced it?"

## What DNAscape Can Be Used For

- **Infer candidate origin activity from replication timing.**  
  Estimate the initiation landscape that could explain a Repli-seq-like timing profile under fixed assumptions about fork speed and genome geometry.

- **Predict replication timing from origin firing.**  
  Start from an origin activity or initiation-rate landscape and compute the expected timing profile produced by bidirectional fork propagation.

- **Compare deterministic mappings with stochastic simulations.**  
  Use fast mapping functions for expected behaviour, then test variability and single-molecule or single-cell effects with ensemble simulations.

- **Model linear and periodic replication domains.**  
  Analyse ordinary chromosomal intervals as well as circular or effectively periodic systems, including ecDNA-inspired domains.

- **Explore fork directionality and fork-based features.**  
  Map between timing gradients and fork directionality, and simulate fork movement, fork encounters, origin efficiencies, inter-origin distances, and S-phase completion times.

- **Test mechanistic scenarios.**  
  Ask how replication changes when origin firing is redistributed, fork speed is altered, forks stall, or circular domain size changes.

- **Bridge genomic data and kinetic interpretation.**  
  Work with replication timing, initiation-like tracks, fork-directionality-like signals, genomic intervals, and gene annotations in a shared modelling workflow.

- **Build reproducible analysis notebooks.**  
  Use compact examples and validation notebooks to move from synthetic toy systems to genome-scale or ecDNA-scale biological case studies.

## Core Workflow

DNAscape is most naturally used as a loop:

```text
assumptions
    -> firing or timing data
    -> mapping / fitting
    -> stochastic simulation
    -> biological features and plots
    -> revised assumptions
```

A typical validation analysis might start with a known firing landscape, predict replication timing, simulate many stochastic realisations, and then test whether the fitting machinery can recover the original firing assumptions from the generated timing profile.

```text
known firing rate
    -> predicted replication timing
    -> simulated replication ensembles
    -> fitted firing rate from timing
    -> comparison with the known input
```

This makes DNAscape useful both as an analysis tool for real data and as a controlled environment for testing hypotheses about replication mechanisms.

## Main Capabilities

- Deterministic mapping from firing rate to replication timing.
- Inference of firing rate from replication timing under fixed assumptions.
- Mapping between replication timing and fork directionality.
- Stochastic ensemble simulation of origin firing and bidirectional fork propagation.
- Linear and periodic domain support for selected mapping and simulation workflows.
- Simulation outputs including replication timing, replicated fraction, fork directionality, origin efficiency, inter-origin distances, number of fired origins, and S-phase duration.
- Optional time-resolved and space-time outputs for richer simulation diagnostics.
- Plotting utilities for line plots, distributions, density plots, heatmaps, and genomic/gene-track style visualisation.
- Data helpers for loading example replication tracks and genomic annotations.
- Notebook-based examples for testing and extending complete analysis pipelines.

## Installation

DNAscape requires Python 3.9 or later.

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import numpy as np
from dnascape import map_firing_timing, map_timing_firing, rsim

# A simple synthetic initiation landscape.
x = np.linspace(-3, 3, 500)
firing_rate = 0.02 + 0.08 * np.exp(-(x ** 2))

# Forward mapping: origin activity -> expected replication timing.
timing = map_firing_timing(
    firing_rate,
    fork_speed=1.4,
    resolution=1.0,
)

# Inverse mapping: replication timing -> inferred origin activity.
recovered_firing_rate = map_timing_firing(
    timing,
    fork_speed=1.4,
    resolution=1.0,
    maxiter=8,
    verbose=False,
)

# Stochastic simulation under the same initiation assumptions.
sim = rsim(
    firing_rate,
    fork_speed=1.4,
    sim_number=50,
    resolution_space=1.0,
    seed=1,
    verbose=False,
)

sim.keys()
```

## Command-Line Interface

After installation, the `dnascape` command is available:

```bash
dnascape --help
```

The command-line interface auto-discovers public DNAscape functions and exposes them as subcommands. For example:

```bash
dnascape rsim --ori-rate frates.npy --fork-speed 1.4 --sim-number 100 --time-statsq
dnascape rfit --d1 firing_rate --d2 replication_timing --source frates.npy
```

For larger analyses, the Python API and notebooks are currently the recommended interface because they make assumptions, intermediate arrays, plots, and validation checks easier to inspect.

## Examples

Example notebooks are available in `examples/`. The intended direction is to build these into a clear validation ladder:

- small synthetic linear domains
- deterministic mapping tests
- stochastic simulation tests
- timing-to-firing recovery tests
- periodic and circular-domain examples
- ecDNA-inspired biological case studies
- roadmap-driven feature tests

## Documentation

Documentation is available at:

- [DNAscape documentation](https://dnascape.readthedocs.io/en/latest/)

The documentation will be expanded alongside the validation notebooks and roadmap work. At this stage, the README and example notebooks are the most direct entry points for understanding the modelling workflow.

## Project Status

DNAscape is under active development. Current work focuses on polishing the full simulation and mapping pipeline, strengthening tests, improving notebook-based validation, and expanding biologically framed documentation.

Planned and in-progress items are tracked in `ROADMAP.md`.

## Citation

If you use DNAscape in your research, please cite the corresponding publication or software release.

## Questions, Bugs, and Feature Requests

Please open an issue:

- https://github.com/fberkemeier/DNAscape/issues

For technical questions or collaboration requests, contact Francisco Berkemeier at fp409@cam.ac.uk.
