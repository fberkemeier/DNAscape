<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape turns DNA replication sequencing data into quantitative kinetic models.

It is built for biologists who measure replication timing, origin activity, fork movement, or replication stress and want to move from genomic tracks to testable mechanisms.

Built on Kolmogorov-Johnson-Mehl-Avrami (KJMA)-style models of replication kinetics, DNAscape provides a common framework for interpreting replication dynamics across genomic space and S-phase time.

DNAscape connects Repli-seq, scRepli-seq, EdU-seq, SNS-seq, Ini-seq, OK-seq, FORK-seq-style profiles, and custom genomic signals to kinetic features that can be inferred, predicted, or simulated: replication timing, origin firing rate, effective fork speed, fork directionality/RFD, replicated fraction, fork density, initiation events, coalescence events, origin efficiency, completion time, and perturbation responses.

## From Sequencing Tracks to Kinetics

DNAscape treats replication assays as different views of the same underlying replication programme. A timing profile, an origin-enrichment profile, a fork-directionality track, or a simulated landscape can each be used as a starting point for model-based inference.

The outputs are not only fitted curves. DNAscape returns interpretable replication features that can be compared across cell types, treatments, perturbations, and candidate mechanisms.

| Starting point | DNAscape can infer or generate |
| --- | --- |
| Replication timing profiles | origin firing, effective fork speed, predicted RFD, replicated fraction |
| Origin-activity assays | timing, origin efficiency, initiation-event density, simulated S phases |
| Fork-directionality or fork-density data | effective fork propagation, coalescence profiles, replication-domain structure |
| User-defined kinetic landscapes | stochastic replication ensembles, completion times, perturbation responses |

The central design principle is that assumptions should be explicit: the same analysis records the data used, the kinetic feature requested, and the biological or numerical constraints imposed.

## Core Workflows

DNAscape has two main workflows: `rmap` for kinetic inference and feature mapping, and `rsim` for stochastic replication simulation.

### `rmap`: Feature Inference and Mapping

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
from dnascape import rmap

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

Inspect available mappings and required inputs:

```python
rmap.targets()
rmap.requirements(target="firing_rate")
```

### `rsim`: Replication Simulation

`rsim` generates stochastic S phases from origin firing and fork-speed assumptions. Use it to compare fitted models with data, explore perturbations, estimate cell-to-cell variability, and generate synthetic replication landscapes.

```python
from dnascape import rsim

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

## Installation

DNAscape requires Python 3.9 or later.

### Python Package

```bash
python -m pip install dnascape
```

### Streamlit App

A Streamlit interface is planned for interactive data upload, visual exploration, simulation, download, zooming, and reproducible code export. Until then, the Python package and notebooks are the main interfaces.

### Build From Source

Use the source install for the latest development version, examples, documentation, and notebooks.

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape

python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

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
