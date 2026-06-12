<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is software for mapping and simulating DNA replication kinetics across genomic space and S-phase time. It connects origin initiation and replication fork dynamics to measurable features of the replication programme, including replication timing, origin activity and fork dynamics, providing a platform for researchers to move between sequencing data, inferred kinetic features and simulated replication programmes within a common mechanistic framework.

Built on Kolmogorov-Johnson-Mehl-Avrami (KJMA)-style models of replication kinetics, DNAscape connects Repli-seq, scRepli-seq, EdU-seq, SNS-seq, Ini-seq, OK-seq, FORK-seq-style profiles, and custom genomic signals to kinetic features that can be inferred, predicted, or simulated: replication timing, origin firing rate, effective fork speed, fork directionality/RFD, replicated fraction, fork density, initiation events, coalescence events, origin efficiency, completion time, and perturbation responses.

## Core functionality

With DNAscape, you can

- **Map** replication features from different data sources, including replication timing, firing rate, fork speed, and related outputs.
- **Simulate** DNA replication programmes in space and time from specified origin initiation and fork speed assumptions.
- **Test perturbations** such as altered initiation, reduced fork speed, fork stalling, dormant-origin activation, or changes in domain structure.
- **Validate pipelines** using synthetic replication programmes where the ground truth is known.
- **Visualise** replication tracks, origin-fork distributions, genomic contexts, among others.

## Installation

DNAscape requires Python 3.9 or later, and is available in 3 different modes.

### Python package

For the standard Python package install:

```bash
pip install dnascape
```

### Build from source

Use the source install for the latest development version, including examples, documentation, and notebooks.

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape

pip install -r requirements.txt
pip install .
```

### Streamlit app

A Streamlit app is planned for browser-based data upload, visual exploration, simulation, download, zooming, and reproducible code export. Until then, use the Python package and example notebooks.

## Workflows

DNAscape has two main workflows: `rmap` for kinetic inference and feature mapping, and `rsim` for stochastic replication simulation.

### `rmap`: feature inference and mapping

`rmap` moves between replication features under stated assumptions. It can infer kinetic quantities from data, predict observables from a proposed model, or translate between compatible experimental readouts.

Examples of `rmap` targets include:

- `firing_rate`
- `fork_speed`
- `replication_timing`
- `fork_directionality`
- `replicated_fraction`
- `replicated_fraction_xt`
- `fork_densities_xt`
- `initiation_events`
- `coalescence_events`
- `origin_efficiency`

```python
firing_rate = rmap(
    source={
        "replication_timing": timing,
        "fork_speed": 1.4,
    },
    target="firing_rate",
    assumptions={
        "resolution_x": 1.0,
        "target_form": "space",
    },
)
```

### `rsim`: replication simulation

`rsim` generates stochastic S phases from origin firing and fork-speed assumptions. Use it to compare fitted models with data, explore perturbations, estimate cell-to-cell variability, and generate synthetic replication landscapes.

```python
simulation = rsim(
    source={
        "firing_rate": firing_rate,
        "fork_speed": 1.4,
    },
    assumptions={
        "sim_number": 100,
        "resolution_x": 1.0,
        "resolution_t": 1.0,
        "seed": 1,
        "verbose": False,
    },
)

replication_timing = simulation["replication_timing"]
```

`rmap` and `rsim` share the same input style:

```text
source       measured data or kinetic inputs
target       requested feature, for rmap
assumptions  resolution, geometry, fitting controls, and biological constraints
```

This makes it natural to infer kinetic fields with `rmap`, pass them into `rsim`, and compare simulated features back to measured data.

## Documentation

Full documentation is available at https://dnascape.readthedocs.io/en/latest/.

Recommended starting points:

- `examples/0-quickstart.ipynb`: first pass through `rmap` and `rsim`
- `examples/2-mapping.ipynb`: mapping and inference atlas
- `examples/3-simulation.ipynb`: stochastic simulation atlas
- `examples/4-neat_examples.ipynb`: compact biological examples

DNAscape treats inputs as regular genomic and time grids. Use `resolution_x` for genomic bin size in kb and `resolution_t` for simulation time step in minutes. If kinetic inputs are constants, provide a genomic length, number of bins, or interval so DNAscape can build the domain.

## Command Line

After installation, a lightweight command-line interface is available:

```bash
dnascape --help
```

For exploratory modelling, the Python API and notebooks are recommended because assumptions, intermediate arrays, plots, and validation checks stay visible.

## Repository Layout

```text
dnascape/      Python package: mapping, simulation, fitting, plotting, IO
examples/      Jupyter notebooks and compact analysis examples
docs/          ReadTheDocs source
data/          small bundled data used by examples
figures/       figures and visual outputs
paper/         manuscript-related material
ROADMAP.md     active development notes
```

## Project Status

DNAscape is developed as a Python package first. The current workflow is the Python API plus notebooks, with a CLI for scripted runs and a planned Streamlit app for interactive use.

## Citation

If you use DNAscape in your research, please cite the corresponding publication or software release.

## Questions and Issues

Please open issues at https://github.com/fberkemeier/DNAscape/issues.

For technical questions or collaboration requests, contact Francisco Berkemeier at fp409@cam.ac.uk.
