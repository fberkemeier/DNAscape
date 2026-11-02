<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape" width="420">
</p>

# DNAscape
[![Documentation Status](https://app.readthedocs.org/projects/dnascape/badge/?version=latest)](https://dnascape.readthedocs.io/en/latest/)

DNAscape is software for mapping and simulating DNA replication, through two complementary workflows.

Its *mapping workflow* infers origin firing and fork speed from data on replication timing, origin competence, fork directionality and other features. Its *simulation workflow* models individual S phases and simulates whole-genome replication programmes from initiation and fork-speed inputs, quantifying inter-origin distances, origin efficiency, S-phase duration and other replication statistics. Together, these workflows can be used to investigate the effects of replication stress, quantify variability between cells, and test hypotheses about the mechanisms shaping replication across organisms and cell types.

DNAscape works with experimental or model-derived profiles of replication timing, origin activity, and fork dynamics, including profiles derived from Repli-seq, SNS-seq and OK-seq. Tools for preparing data, visualising results and analysing datasets support both workflows. Region masking and transformation tools allow users to test local perturbations and examine discrepancies between predicted and observed replication profiles in selected genomic regions. Use DNAscape through its browser app, in Python scripts and Jupyter notebooks, or from the command line.

## Run the DNAscape app

DNAscape is best used through its user-friendly, interactive app, built with [NiceGUI](https://nicegui.io). To run the app locally, follow the steps below.

> **Requirements:** Python 3.10 or newer (3.13 is recommended), Git, Pip and a modern web browser.

### 1. Download the repository

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape
```

### 2. Install and run the app

```bash
pip install .[app]
python -m app
```

## A quick first run

Here we present a quickstart example for processing data, mapping a kinetic feature, and simulating a replication programme.

### I. Managing data

With the app open, select **Data Manager** in the left-hand menu. We will prepare a replication-timing profile from publicly available Repli-seq data:

1. Download the [K562 Repli-seq bigWig file](https://hgdownload.gi.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig) from ENCODE and **Upload** it. Alternatively, choose **Link**, select the K562 Repli-seq example to automatically paste the ENCODE link and click **Inspect**. This uploads the raw bigWig file onto DNAscape.
2. In **Configure**, set **Prepare as** to `Replication timing`. Select `chr1` and set **Bin size (bp)** to `1000`. Under **Values**, set **Missing values** as `Fill with a value` and set **Fill value** to `30` (minutes).
3. In **Data transforms**, switch on **Apply transform** and leave the formula as `source`. Set **Value scaling** to `Target interval`, **Minimum** to `30`, **Maximum** to `500`, and check **Invert**, so higher Repli-seq scores represent earlier replication. This assigns an illustrative 30–500 minute range to relative timing scores.
4. Click **Convert**, inspect the plot, and download the timing profile from **Generated Files → Raw data** as TSV.

### II. Mapping features

Select **Mapping** from the left-hand menu. We will now infer a spatial origin initiation-rate profile that fits the timing data:

1. Within **Target**, set **Mapped output** to `Initiation rate`, **Dependence** to `Space`, and **Input factors** to `Replication timing + fork speed`. This determines the mapping route: to determine the initation rate from given replication timing and fork speed.
2. Under **Inputs**, upload the timing profile as a spatial profile. Set **Fork speed** to `Constant` with a value of `1.4` kb/min.
3. In **Region**, leave **Whole region** off and set **Start** to `40000` kb, **End** to `50000` kb, and keep **Resolution** at `1` kb. Click **Map**, compare observed and fitted timing (toggle the **Output** options in the **Preview** panel), and download the fitted initiation-rate profile from **Raw data**.

### III. Simulating DNA replication

Move to **Simulation** to simulate replication using the inferred origin-firing profile:

1. Under **Initiation rate**, choose `Profile` and upload the mapped profile. DNAscape automatically detects the space dependence. Keep **Fork speed** constant at `1.4` kb/min.
2. In **Region**, enable **Whole region** and keep **Resolution** at `1` kb/bin. In **Simulation settings**, set **Simulations (replicates)** to `100`; optionally enable **Fix seed** for reproducible runs.
3. Click **Simulate**, explore the outputs in **Preview**, and download data or figures. Change the fork speed, number of simulations or other settings and rerun to compare outcomes.

## Using the source code

DNAscape is written in Python. For large-scale simulations, automated analyses or customised workflows, you can export code from the app or use its Python modules directly.

### Export a workflow from the app

The app includes an option to directly download the relevant code for one of DNAscape's core modules, namely **Data Manager**, **Mapping**, **Simulation** and **Analysis**. Enabling **Settings → Interface → Developer mode** displays a **Code** button next to the action button in each page. Each button exports a self-contained Python script reproducing the page’s workflow with your current settings and input/output folder definitions. Run the script with DNAscape installed and the required files in its input folder. Uploads are optional: missing inputs use placeholder filenames that you can edit to reference local data.

### Use the Python modules directly

An alternative way is to use the more user-friendly DNAscape’s modules into your own scripts or notebooks. Within `examples`, see `1-quickstart.ipynb`, `2-mapping.ipynb` and `3-simulation.ipynb` for worked examples.

## Documentation

Please see the [documentation](https://dnascape.readthedocs.io/en/latest) for detailed usage instructions, descriptions of DNAscape's features, and further examples.

## Citation

If you use DNAscape in your research, please cite the following and include the software version used:

> Francisco Berkemeier and Michael A. Boemo (2026). DNAscape: Software for mapping and simulating DNA replication kinetics. bioRxiv.

## Questions and bugs

For questions, bug reports or feature requests, use [GitHub Issues](https://github.com/fberkemeier/DNAscape/issues) or contact [Francisco Berkemeier](mailto:fp409@cam.ac.uk).

DNAscape is available under the [MIT License](LICENSE).
