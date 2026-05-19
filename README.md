<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is software for fitting, mapping, and simulating DNA replication kinetics from genomic data.

The model represents replication as a kinetic process defined by an initiation rate and a fork speed that may depend on genomic position and time. Under explicit assumptions about these quantities, DNAscape connects latent replication mechanisms to observable data types such as replication timing, origin activity, fork directionality, replicated fraction, fork density, coalescence events, origin efficiency, and completion time.

DNAscape can be used in two complementary directions:

- **Forward modelling**: start from a mechanistic hypothesis, such as a proposed initiation landscape or fork-speed profile, and predict the replication programme it would generate.
- **Inverse mapping**: start from observed replication data, such as a timing profile, and infer the kinetic quantities that could explain it under fixed assumptions.

This makes DNAscape useful for analysing experimental datasets, generating synthetic controls, comparing biological conditions, and testing mechanistic hypotheses about how replication is regulated or perturbed.

## Scientific scope

DNAscape is built around a general replication-kinetic description:

```text
initiation rate: I(x, t)
fork speed:      v(x, t)
```

where `x` is genomic position and `t` is replication time. Individual analyses can restrict this general description to simpler cases, such as constant fork speed, spatially varying initiation, periodic domains, or stochastic origin firing.

The goal is not only to reproduce a measured profile, but to make assumptions explicit enough that different biological explanations can be compared.

## Applications

- **Fit latent initiation programmes from replication timing.**  
  Infer candidate origin firing landscapes that explain Repli-seq-like timing profiles under specified assumptions about fork speed, geometry, and boundary conditions.

- **Predict observables from mechanistic hypotheses.**  
  Simulate or map how a proposed initiation rate, fork-speed field, or perturbation would appear in timing, fork directionality, replicated fraction, fork density, and completion-time outputs.

- **Compare datasets in a common kinetic language.**  
  Relate timing, initiation-enriched assays, fork-directionality signals, and simulated outputs through shared quantities rather than treating each track as an isolated signal.

- **Test perturbation and therapeutic scenarios.**  
  Ask how replication changes when initiation is redistributed, fork speed is reduced, forks stall, dormant origins compensate, or replication stress alters completion dynamics.

- **Separate model assumptions from biological conclusions.**  
  Repeat fits and simulations under alternative fork speeds, domain geometries, stochastic rules, or stress models to identify which conclusions are robust.

- **Generate synthetic benchmarks and controls.**  
  Create known replication programmes, observe them through simulated readouts, and test whether fitting or downstream analysis recovers the expected mechanism.

- **Analyse variability across molecules, cells, or simulations.**  
  Use stochastic ensembles to quantify origin efficiency, inter-origin distances, fork encounters, replicated fraction through time, and S-phase completion distributions.

- **Integrate replication with genomic context.**  
  Combine replication features with genomic intervals, gene annotations, and region-based analyses for biologically interpretable visualisation.

## Core workflow

DNAscape is most naturally used as an iterative modelling workflow:

```text
biological question
    -> explicit kinetic assumptions
    -> mapping or stochastic simulation
    -> predicted replication features
    -> comparison with data
    -> revised hypothesis
```

A controlled validation workflow can start from a known mechanism and ask whether the software recovers it:

```text
known initiation rate and fork speed
    -> predicted timing and fork features
    -> stochastic replication ensembles
    -> fitted kinetic quantities from simulated observations
    -> comparison with the known input
```

This forward-and-inverse structure is intended to support both exploratory biological modelling and rigorous pipeline testing.

## Main capabilities

- Deterministic mapping from firing rate to replication timing.
- Inference of firing rate from replication timing under fixed assumptions.
- Mapping between replication timing and fork directionality.
- Stochastic ensemble simulation of origin firing and bidirectional fork propagation.
- Support for linear and periodic replication domains in selected workflows.
- Spatial and time-resolved simulation inputs for initiation rate and fork speed.
- Simulation of fork stalling, heterogeneous fork behaviour, and replication-stress scenarios.
- Outputs including replication timing, replicated fraction, fork directionality, origin efficiency, inter-origin distances, number of fired origins, fork densities, coalescence events, and S-phase duration.
- Plotting utilities for line plots, distributions, density plots, heatmaps, and genomic or gene-track visualisation.
- Data helpers for loading example replication tracks and genomic annotations.
- Notebook-based workflows for reproducible modelling, validation, and biological case studies.

## Installation

DNAscape requires Python 3.9 or later.

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
import numpy as np
from dnascape import map_firing_timing, map_timing_firing, rsim

# Define a simple synthetic initiation landscape.
x = np.linspace(-3, 3, 500)
firing_rate = 0.02 + 0.08 * np.exp(-(x ** 2))

# Forward mapping: initiation activity -> expected replication timing.
timing = map_firing_timing(
    firing_rate,
    fork_speed=1.4,
    resolution=1.0,
)

# Inverse mapping: replication timing -> inferred initiation activity.
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

## Command-line interface

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

- small synthetic domains
- deterministic mapping tests
- stochastic simulation tests
- timing-to-firing recovery tests
- periodic-domain examples
- biological case studies
- roadmap-driven feature tests

## Documentation

Documentation is available at:

- [DNAscape documentation](https://dnascape.readthedocs.io/en/latest/)

The documentation will be expanded alongside the validation notebooks and roadmap work. At this stage, the README and example notebooks are the most direct entry points for understanding the modelling workflow.

## Project status

DNAscape is under active development. Current work focuses on polishing the full simulation and mapping pipeline, strengthening tests, improving notebook-based validation, and expanding biologically framed documentation.

Planned and in-progress items are tracked in `ROADMAP.md`.

## Citation

If you use DNAscape in your research, please cite the corresponding publication or software release.

## Questions, bugs, and feature requests

Please open an issue:

- https://github.com/fberkemeier/DNAscape/issues

For technical questions or collaboration requests, contact Francisco Berkemeier at fp409@cam.ac.uk.
