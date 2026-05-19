<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is a Python package for fitting, mapping, and simulating DNA replication kinetics from genomic data.

At its core, DNAscape represents genome replication as origin initiation followed by bidirectional fork propagation. Analyses can define or infer an initiation rate `I(x, t)` and fork speed `v(x, t)`, where both quantities may vary across genomic position and replication time. This allows measured replication features to be interpreted in terms of explicit kinetic hypotheses.

DNAscape is designed for researchers who want to connect replication timing, origin activity, fork movement, and completion dynamics in a single reproducible workflow.

## Core functionality

- **Fit** initiation programmes from observed replication timing under stated assumptions.
- **Map** between firing rate, replication timing, fork directionality, and related replication features.
- **Simulate** stochastic replication ensembles from specified initiation and fork-speed landscapes.
- **Test** how changes in initiation, fork speed, fork stalling, or domain structure affect observable profiles.
- **Validate** analysis pipelines with synthetic replication programmes where the ground truth is known.
- **Visualise** replication tracks, distributions, heatmaps, and genomic feature context.

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

# Define a synthetic initiation landscape.
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

Example notebooks are available in `examples/`. These are being organised into a validation ladder:

- small synthetic domains
- deterministic mapping tests
- stochastic simulation tests
- timing-to-firing recovery tests
- periodic-domain examples
- biological case studies
- roadmap-driven feature tests

## Scientific scope

DNAscape is built around a general replication-kinetic description:

```text
initiation rate: I(x, t)
fork speed:      v(x, t)
```

where `x` is genomic position and `t` is replication time. Simple analyses can use fixed fork speed and position-dependent initiation. More general analyses can allow both initiation and fork speed to vary across space and time.

This supports two complementary modes of analysis:

- **Forward modelling** starts from a proposed mechanism and predicts the replication profiles it would generate.
- **Inverse mapping** starts from observed replication data and infers the kinetic quantities that could explain it under fixed assumptions.

The purpose is not only to reproduce measured tracks, but to make the assumptions behind each interpretation explicit, testable, and comparable.

## Applications

- **Interpret replication timing.**  
  Infer candidate initiation landscapes that explain timing profiles under explicit fork-speed, geometry, and boundary assumptions.

- **Connect replication observables.**  
  Relate timing, initiation activity, fork directionality, replicated fraction, and simulation outputs through common kinetic quantities.

- **Test mechanistic hypotheses.**  
  Ask whether altered initiation, reduced fork speed, fork stalling, dormant-origin activation, or other perturbations can explain an observed phenotype.

- **Benchmark analysis methods.**  
  Generate synthetic replication programmes, observe them through simulated outputs, and test whether fitting or downstream analysis recovers the known mechanism.

- **Integrate replication with genomic context.**  
  Combine replication features with genomic intervals, gene annotations, and region-based plots for biologically interpretable analysis.

## Workflow

A typical DNAscape analysis is an explicit loop between biological assumptions and observable replication features:

```text
biological question
    -> kinetic assumptions
    -> mapping or stochastic simulation
    -> predicted replication features
    -> comparison with data
    -> revised hypothesis
```

A validation workflow can start from a known mechanism and test whether the pipeline recovers it:

```text
known initiation rate and fork speed
    -> predicted timing and fork features
    -> stochastic replication ensembles
    -> fitted kinetic quantities from simulated observations
    -> comparison with the known input
```

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
