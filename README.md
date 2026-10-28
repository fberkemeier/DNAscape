<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape" width="420">
</p>

# DNAscape

DNAscape is software for mapping and simulating DNA replication kinetics. It infers origin firing and fork movement from genomic data and predicts replication timing and other replication features. Use it through Python, notebooks, the command line or the local browser app.

## What you can do

- **Prepare experimental data:** derive replication-timing profiles from Repli-seq, origin-firing profiles from SNS-seq and fork-directionality profiles from OK-seq. Select genomic regions, convert resolutions, smooth and rescale signals, or transform selected regions.
- **Map replication features:** choose a target, such as initiation rate, fork speed, replication timing or fork directionality, and infer it from compatible input profiles or constant values. Fit spatial, temporal or space-time dependence where supported.
- **Simulate DNA replication:** model origin firing and fork movement, explore changes in initiation rate or fork speed, and inspect timing, fork profiles, replicated fractions and origin statistics across simulations.
- **Visualise and analyse results:** compare datasets, explore profiles and distributions, perform statistical analyses and customise plots. Download prepared data and model outputs as TSV, TXT, CSV or NPZ, alongside figures and analysis reports.

## Run the DNAscape app

An online version is a work in progress. To run the app locally, follow the steps below.

### Requirements

- Python 3.10 or newer (Python 3.13 recommended)
- Git
- A modern web browser

### 1. Download the repository

```bash
git clone https://github.com/fberkemeier/DNAscape.git
cd DNAscape
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **BigWig note:** if you use BigWig files or links, installation automatically selects `pybigtools` for Python 3.10–3.13 or `figwig` for Python 3.14 and newer. Both provide the same region and binning controls. Figwig's first read compiles and caches its decoding routines, so it can take a few seconds longer.

### 3. Install the app

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt
```

### 4. Start DNAscape

```bash
python -m app
```

## A quick first run

### Managing data

Start in **Data Manager** to prepare a replication-timing profile:

1. Upload a Repli-seq track, or choose **Link**, paste the [K562 Repli-seq BigWig link](https://hgdownload.gi.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig) and click **Inspect**.
2. In **Configure**, set **Prepare as** to **Replication timing**. Select **chr1**, keep the full chromosome, and set **Bin size (bp)** to **1000** and **Aggregation** to **Mean**. Under **Values**, choose **Missing values → Fill with a value** and set **Fill value** to **30**.
3. In **Data transforms**, switch on **Apply transform** and leave the formula as `source`. Set **Value scaling** to **Target interval**, **Minimum** to **30**, **Maximum** to **500**, and check **Invert**, so higher Repli-seq scores represent earlier replication. This assigns an illustrative 30–500 minute range to relative timing scores.
4. Click **Convert**, inspect the plot, and download the timing profile from **Generated Files → Raw data** as TSV.

### Mapping data

Move to **Mapping** to infer a spatial initiation-rate profile from that timing data:

1. Set **Mapped output** to **Initiation rate**, **Dependence** to **Space**, and **Input factors** to **Replication timing + fork speed**.
2. Under **Inputs**, upload the timing profile as a spatial profile. Set **Fork speed** to **Constant** with a value of **1.4 kb/min**.
3. In **Region**, leave **Whole region** off and set **Start** to **40000 kb**, **End** to **50000 kb**, and **Resolution** to **1 kb**. Click **Map**, compare observed and fitted timing, and download the fitted initiation-rate profile from **Raw data**.

### Simulating DNA replication

Move to **Simulation** to simulate replication using the inferred origin-firing profile:

1. Under **Initiation rate**, choose **Profile** and upload the mapped profile with **Space** dependence. Keep **Fork speed** constant at **1.4 kb/min**.
2. In **Region**, enable **Whole region** and keep **Resolution** at **1 kb/bin**. In **Simulation settings**, set **Simulations (replicates)** to **100**; optionally enable **Fix seed** for reproducible runs.
3. Click **Simulate**, explore the outputs in **Preview**, and download data or figures. Change the fork speed, number of simulations or other settings and rerun to compare outcomes.

For advanced examples and further workflows, see the [documentation](https://dnascape.readthedocs.io/en/latest/examples.html), which is still being expanded.

## Using the Python code

Use the [Python API](https://dnascape.readthedocs.io/en/latest/api_reference.html) in scripts or notebooks; install the package from this checkout with `python -m pip install .`.

To turn an app workflow into code, enable **Settings → Interface → Developer mode**, then click **Code** on Data Manager, Mapping, Simulation or Analysis. This downloads an editable `.py` script with the current settings and input/output folder definitions. Place any listed source files in its input folder and run it with DNAscape installed.

## Citation

If you use DNAscape in your research, please cite the following and include the software version used:

> Francisco Berkemeier and Michael A. Boemo (2026). DNAscape: Software for mapping and simulating DNA replication kinetics. bioRxiv.

## Questions and bugs

For questions, bug reports or feature requests, use [GitHub Issues](https://github.com/fberkemeier/DNAscape/issues) or contact [Francisco Berkemeier](mailto:fp409@cam.ac.uk).

DNAscape is available under the [MIT License](LICENSE).
