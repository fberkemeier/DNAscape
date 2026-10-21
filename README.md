<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape" width="420">
</p>

# DNAscape

DNAscape is a local browser app for preparing, mapping, simulating, visualising, and analysing DNA replication data. It brings the main DNAscape workflows into one interface while keeping the computation on your computer.

## What you can do

- **Prepare data:** import tables, NumPy profiles, bedGraph, or BigWig tracks; select genomic regions; bin, smooth, rescale, invert, or filter values; and export DNAscape-ready profiles.
- **Map replication features:** infer supported kinetic targets from one or more compatible input profiles or constant values.
- **Run simulations:** simulate replication from firing-rate and fork-speed profiles or constants, compare datasets, and inspect standard or detailed outputs.
- **Plot and analyse data:** compare multiple profiles, smooth each dataset independently, export plots, and calculate Pearson, Spearman, and Kendall tau correlations. Download all selected correlation matrices and paired sample counts in one CSV or TSV report.
- **Reuse recent work:** reopen or download files and completed workflows retained in the current browser session.

The app accepts common profile formats including CSV, TSV, TXT, NPY, NPZ, bedGraph, and BigWig. The Data Manager can load supported files from public links and import GEO 16-fraction Repli-seq text matrices (`.mat` or `.mat.gz`), including chromosome-specific replication timing extraction.

Raw data exports support TSV, TXT, CSV and NPZ. Analysis reports use CSV or TSV.

## Run locally

### Requirements

- Python 3.10 or newer (Python 3.13 recommended for BigWig imports)
- Git
- A modern web browser

The app installs its BigWig reader on Python 3.10–3.13. On Python 3.14 or
newer, tables, NumPy profiles, and bedGraph tracks work; BigWig imports require
a separately installed compatible reader. If you need BigWig support, install
Python 3.13 (`winget install -e --id Python.Python.3.13`) and select it when creating the environment with
`py -3.13 -m venv .venv` on Windows or `python3.13 -m venv .venv` on macOS/Linux.

### 1. Download the repository

```bash
git clone https://github.com/fberkemeier/DNAscape-devmode.git
cd DNAscape-devmode
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

### 3. Install the app

Run these commands from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt
```

### 4. Start DNAscape

```bash
python -m app
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser. Keep the terminal open while using the app, and press `Ctrl+C` there to stop it.

## A quick first run

1. Open **Simulation** from the left menu.
2. Keep the default constant inputs and select **Simulate**.
3. Explore the generated outputs and plotting settings in the Preview panel.

To try a file-based workflow, open **Data Manager** and use the bundled example at `examples/data/example/replication_timing_hESC.csv`.

## Managing data

Use **Data Manager** to turn a Repli-seq BigWig track into a replication-timing profile. This example prepares 1 kb bins with timings from **30 to 500 minutes**, spanning approximately eight hours.

1. In **Data**, upload or drag in a BigWig file. Alternatively, open **Link**, select **Example: ENCODE Repli-seq (bigWig) - K562**, then click **Inspect**. You can also paste your own public BigWig URL and inspect it.
2. In **Configure**, set **Prepare as** to **Replication timing**. Under **Region & binning**, choose **One chromosome**, select the chromosome and Start/End coordinates, set **Bin size (bp)** to **1000**, and keep **Aggregation** at **Mean**.
3. Under **Values**, set **Value scaling** to **Target interval**, **Minimum** to **30**, and **Maximum** to **500**. Check **Invert** so higher Repli-seq scores become earlier replication times.
4. Set **Missing values** to **Fill with a value**, and **Fill value** to **30**. Custom fills are applied after scaling, so missing bins become exactly 30 minutes.
5. Under **Output**, optionally enable **Show plot preview after conversion**, then click **Convert**. Download the result from **Generated files**; TSV is the default format. When using the profile in Mapping or Simulation, set the spatial resolution to **1 kb** to match the 1000 bp bins.

The K562 example contains relative timing scores; rescaling them to 30–500 minutes assigns an assumed absolute time range. Filling missing bins with 30 assigns them the earliest time in this example. Other missing-value choices are **Keep empty values**, **Fill with zero** (the default), **Fill with minimum**, **Fill with maximum**, and **Report error**.

## Comparing and analysing data

Open **Analysis** and upload the profiles you want to compare. Use matching genomic regions and resolutions: correlations pair samples by array index over their shared span. Switch on **Enable analysis** to reveal correlation choices and independent moving-average or Gaussian smoothing for each dataset. **Show p-values** optionally displays two-sided, unadjusted p-values for dataset pairs.

Choose Pearson, Spearman, and/or Kendall tau, then click **Analyse**. The **Analysis report** below the plotting settings combines all selected methods in one table, with each correlation heading its dataset columns. At the bottom, edit the report name, choose CSV (default) or TSV, and use the download arrow to save the coefficients, paired sample counts, and optional p-values. The cross clears the plots and report while retaining uploaded data. With analysis switched off, **Plot** uses the original profiles and the analysis controls and report are hidden.

## Local data and sessions

DNAscape does not require a hosted service. Uploaded files and calculations are handled by the Python process running on your computer. A public URL is accessed only when you explicitly choose it as a data source.

Recent uploads and results are cached in memory for the current browser tab and are not permanent storage. Use **Cached Files** to manage them, or **Reset to defaults** in the top-right menu to clear the workspace.

## Troubleshooting

- **The page did not open automatically:** visit [http://127.0.0.1:8080](http://127.0.0.1:8080) manually.
- **Port 8080 is already in use:** set `DNASCAPE_APP_PORT` to another port before running `python -m app`.
- **A change is not visible:** stop the app with `Ctrl+C`, start it again, and refresh the browser.
- **PowerShell blocks environment activation:** skip activation and run `.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt`, followed by `.\.venv\Scripts\python.exe -m app`.
- **Installation reports `No matching distribution found for gwseq-io-pp`:** update this checkout and rerun the install command. DNAscape no longer requires that discontinued package.
- **BigWig import reports that no reader is installed:** use Python 3.13 and install the app dependencies in that environment, or import the track as bedGraph.

## Project status

The local interface is under active development. Please report bugs or suggest improvements through [GitHub Issues](https://github.com/fberkemeier/DNAscape/issues).

DNAscape is available under the [MIT License](LICENSE).
