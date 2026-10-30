<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape" width="420">
</p>

# DNAscape

DNAscape is software for mapping and simulating DNA replication. Its mapping workflow infers origin firing and fork speed from data on replication timing, origin competence, fork directionality and other features. Its simulation workflow models individual S phases and simulates whole-genome replication programmes from initiation and fork-speed inputs, quantifying inter-origin distances, origin efficiency, S-phase duration and other replication statistics. Together, the two workflows can be used to investigate the effects of replication stress, quantify variability between cells, and test hypotheses about the mechanisms shaping replication programmes across organisms and cell types.

DNAscape works with experimental or model-derived profiles of replication timing, origin activity, and fork dynamics, including profiles derived from Repli-seq, SNS-seq and OK-seq. Tools for preparing data, visualising results and analysing datasets support both workflows. Use DNAscape through its local browser app, in Python scripts and Jupyter notebooks, or from the command line.

## Run the DNAscape app

To run the app locally, follow the steps below.

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

Run these commands from the project folder. Choose an installed Python version below (`3.13` is recommended):

On Windows PowerShell:

```powershell
$pythonVersion = "3.13"
py "-$pythonVersion" -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS and Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install the app

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt
```

> **BigWig note:** if you use BigWig files, installation automatically selects `pybigtools` for Python 3.10–3.13 or `figwig` for Python 3.14 and newer. Both provide the same region and binning controls.

### 4. Start DNAscape

```bash
python -m app
```

For later launches, activate `.venv` again and run `python -m app`.

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

For advanced examples and further workflows, see the [documentation](https://dnascape.readthedocs.io/en/latest/python.html#notebooks), which is still being expanded.

## Using the Python code

Use the [Python guide](https://dnascape.readthedocs.io/en/latest/python.html) in scripts or notebooks; install the package from this checkout with `python -m pip install .`.

To turn an app workflow into code, enable **Settings → Interface → Developer mode**, then click **Code** on Data Manager, Mapping, Simulation or Analysis. This downloads an editable `.py` script with the current settings and input/output folder definitions. Place any listed source files in its input folder and run it with DNAscape installed.

Uploads are optional for code export: missing inputs use placeholder filenames,
which you can edit to point to data already stored locally.

## Citation

If you use DNAscape in your research, please cite the following and include the software version used:

> Francisco Berkemeier and Michael A. Boemo (2026). DNAscape: Software for mapping and simulating DNA replication kinetics. bioRxiv.

## Questions and bugs

For questions, bug reports or feature requests, use [GitHub Issues](https://github.com/fberkemeier/DNAscape/issues) or contact [Francisco Berkemeier](mailto:fp409@cam.ac.uk).

DNAscape is available under the [MIT License](LICENSE).
