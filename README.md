<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape logo" width="50%">
</p>

# DNAscape

[![Documentation Status](https://readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is software for building quantitative models of DNA replication from genomic sequencing data. It maps measured replication tracks to kinetic quantities, such as timing, origin activity, and fork dynamics, simulates replication programmes from mechanistic assumptions, and compares predicted features across genomic position and S-phase time.

DNAscape is particularly designed for multimodal replication data, including: Repli-seq, BrdU-seq, EdU-seq, SNS-seq, ini-seq, bubble-seq, OK-seq, FORK-seq, DNAscent-style nanopore profiles, single-cell variants of these approaches, and custom genomic tracks.

By keeping data, kinetic inputs, and modelling assumptions explicit, DNAscape is a useful tool to test how perturbations in origin initiation, fork movement, genome geometry, or replication stress shape the replication programme across organisms and cell types.

## Core functionality

With DNAscape, you can

- **Infer kinetic features** from replication timing, origin activity, fork directionality, and related genomic tracks.
- **Simulate replication programmes** from specified origin firing and fork-speed assumptions.
- **Predict measurable outputs** such as replication timing, replicated fraction, fork densities, initiation events, coalescence events, and completion times.
- **Test perturbations** including altered initiation, reduced fork speed, fork stalling, dormant-origin activation, and changes in domain structure.
- **Benchmark analyses** with synthetic replication programmes where the ground truth is known.
- **Visualise outputs** as genomic tracks, time courses, origin-fork distributions, and related summaries.

## Installation

DNAscape requires Python 3.9 or later and can be installed in three ways.

### Python package

For the standard Python package install:

```bash
pip install dnascape
````

### Build from source

Use the source install for the latest development version, including examples, documentation, and notebooks.

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape

pip install -r requirements.txt
pip install .
```

### NiceGUI interface prototype

The new NiceGUI prototype mirrors the DNAscape workflow, visual style, pages,
forms, and controls. Its Simulation page now runs bounded scalar or uploaded
kinetic fields through `rsim` in a background worker and displays live progress
and a result plot. Its optional detailed mode adds temporal and space-time
outputs. The Mapping page resolves the public `rmap` route catalogue, inspects
profile uploads automatically, runs the selected route in a background worker,
and previews the mapped output. Plot and selected raw-data downloads are
generated lazily. The prototype requires Python 3.10 or newer.

```bash
pip install -r app/requirements.txt
python -m app
```

## Workflows

DNAscape is organised around two main workflows: `rmap` for kinetic inference and feature mapping, and `rsim` for stochastic replication simulation.

### `rmap`: feature inference and mapping

`rmap` maps one replication feature to another under stated assumptions. Use it to infer kinetic quantities from data, predict observables from a proposed model, or translate between compatible experimental readouts.

Examples of `rmap` targets include:

* `firing_rate`
* `fork_speed`
* `replication_timing`
* `fork_directionality`
* `replicated_fraction`
* `replicated_fraction_xt`
* `fork_densities_xt`
* `initiation_events`
* `coalescence_events`
* `origin_efficiency`

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

`rmap` and `rsim` use the same input style:

```text
source       measured data or kinetic inputs
target       requested feature, for rmap
assumptions  resolution, geometry, fitting controls, and biological constraints
```

This makes it natural to infer kinetic fields with `rmap`, pass them into `rsim`, and compare simulated outputs back to measured data.

## Documentation

Full documentation is available at https://dnascape.readthedocs.io/en/latest/.

Recommended starting points:

* `examples/0-quickstart.ipynb`: first pass through `rmap` and `rsim`
* `examples/2-mapping.ipynb`: mapping and inference atlas
* `examples/3-simulation.ipynb`: stochastic simulation atlas
* `examples/4-neat_examples.ipynb`: compact biological examples

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
app/           NiceGUI interface prototype
examples/      Jupyter notebooks and compact analysis examples
docs/          ReadTheDocs source
data/          small bundled data used by examples
```

## Project Status

DNAscape is developed as a Python package first. The repository includes the
Python API, notebooks, a CLI for scripted runs, a NiceGUI interface prototype,
and its automated tests and documentation.

## Citation

If you use DNAscape in your research, please cite the corresponding publication or software release.

## Questions and Issues

Please open issues at https://github.com/fberkemeier/DNAscape/issues.

For technical questions or collaboration requests, contact Francisco Berkemeier at [fp409@cam.ac.uk](mailto:fp409@cam.ac.uk).
