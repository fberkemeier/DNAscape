<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape" width="420">
</p>

# DNAscape

DNAscape is a local browser app for preparing, mapping, simulating, visualising, and analysing DNA replication data. It brings the main DNAscape workflows into one interface while keeping the computation on your computer.

**Initiation rate** (`initiation_rate`, abbreviated `ir`) describes the rate of origin firing in unreplicated DNA.

## What you can do

- **Prepare data:** import tables, NumPy profiles, bedGraph, or BigWig tracks; bin, smooth, rescale or invert values; apply region transforms within a mask; and export DNAscape-ready profiles.
- **Map replication features:** infer supported kinetic targets from one or more compatible input profiles or constant values.
- **Run simulations:** simulate replication from initiation-rate and fork-speed profiles or constants, compare datasets, and inspect standard or detailed outputs.
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

Simulation's **Region** includes **Whole region**, **Periodic** and **Show plot** switches. Whole region uses the genomic extent of an uploaded initiation-rate or fork-speed profile. **Region transforms** can edit initiation rate, fork speed, or both independently, after selecting and resampling the region. Each input has its own mask, transform and optional position/interval export. Plot settings provide separate mask guides and names. Gaussian noise uses the transform's seed when supplied, otherwise the simulation seed when fixed. Original uploads are preserved.

To try a file-based workflow, open **Data Manager** and use the bundled example at `examples/data/example/replication_timing_hESC.csv`.

## Managing data

Use **Data Manager** to turn a Repli-seq BigWig track into a replication-timing profile. This example prepares 1 kb bins with timings from **30 to 500 minutes**, spanning approximately eight hours.

The **Region transforms** section defines a mask of selected positions and applies an operation within it. **Invert mask** selects the complement within Start/End; **Export mask** downloads the selected positions or intervals. Configure and **Data transforms** run before these regional operations.

1. In **Data**, upload or drag in a BigWig file. Alternatively, open **Link**, select **Example: ENCODE Repli-seq (bigWig) - K562**, then click **Inspect**. You can also paste your own public BigWig URL and inspect it.
2. In **Configure**, set **Prepare as** to **Replication timing**. Under **Region & binning**, choose **One chromosome**, select the chromosome and Start/End coordinates, set **Bin size (bp)** to **1000**, and keep **Aggregation** at **Mean**.
3. Under **Values**, set **Value scaling** to **Target interval**, **Minimum** to **30**, and **Maximum** to **500**. Check **Invert** so higher Repli-seq scores become earlier replication times.
4. Set **Missing values** to **Fill with a value**, and **Fill value** to **30**. Custom fills are applied after scaling, so missing bins become exactly 30 minutes.
5. Under **Output**, optionally enable **Show plot preview after conversion**, then click **Convert**. Download the result from **Generated files**; TSV is the default format. Mapping and Simulation assume array uploads are sampled at **1 kb**. Their **Resolution** control sets the output bin width: wider bins average the source values, while finer bins use linear interpolation. The selected Start/End interval stays fixed. Initiation rates are expressed per output bin (min⁻¹); fork speeds remain in kb/min, with travel time scaled by bin width.

The K562 example contains relative timing scores; rescaling them to 30–500 minutes assigns an assumed absolute time range. Filling missing bins with 30 assigns them the earliest time in this example. Other missing-value choices are **Keep empty values**, **Fill with zero** (the default), **Fill with minimum**, **Fill with maximum**, and **Report error**.

In Mapping, **Fitting** lists only the target quantity. Set lower and upper bounds independently, or leave a field blank for the route's automatic default. An optional initial guess sets the solver's starting value. Weibull routes fit an initiation-rate amplitude, so their bounds and guess apply to that amplitude. Supplied timing and kinetic profiles remain unchanged. Standard kinetic fits and forward timing/directionality maps use deterministic predictions at zero stall rate; nonzero stalling uses stochastic ensembles. Explicitly stochastic fork-speed and firing-factor models retain their requested simulation model. Periodic joins are used consistently across speed dependences. Timing-gradient directionality requires timing data compatible with the periodic join. Automatic raw-output filenames end in `_c`, `_x`, `_t` or `_xt` for their declared dependence.

Temporal fork-speed fits anchor the initial speed at the first time point and scale their time dependence to the observed timing interval. Sampling timing observations retains the full spatial firing integral, including narrow origins between observation sites. For spatial firing inputs, automatic time grids follow the data and extend until the omitted survival integral is negligible. Explicit time grids are preserved.

Mapping offers deterministic observables. Directionality accepts timing plus fork speed, or initiation rate plus fork speed through the timing relation. Temporal velocities weight the survival gradient before time integration. Directionality from fork densities, initiation/coalescence probability densities and ensemble-only outputs remain available through Simulation and the Python API. Forward time limits are inferred from the kinetic profiles and extended to completion; supplied time grids retain their explicit limits. Replicated fraction integrates partially covered spatial bins, including spatially varying fork travel times.

For **Initiation events** and **Coalescence events**, **Quantity** selects **Density (events/kb)**, integrated over the computed time window, or **Rate density (events/kb/min)**. These expected event densities retain their counts; they are not normalized probability distributions. With unreplicated fraction `s = 1-f`, the rates are `j_i = I*s` and `j_c = 2*v*rho_right*rho_left/s` for symmetric fork speeds, following [Gauthier, Norio and Bechhoefer (2012), Eqs. 1–6](https://doi.org/10.1371/journal.pone.0032053). Both forward outputs use the same deterministic transport solution for full space–time initiation-rate and speed profiles. The solver evolves conditional fork densities `g = rho/s` and `-log(s)` to avoid division near completion, with second-order spatial transport and adaptive time integration. Integrated densities are accumulated during the solve, independently of the displayed time samples. Spatial refinement reduces the remaining discretization error. The terminology follows the unnormalized event densities in [Baker and Bechhoefer (2014), Section II.A](https://doi.org/10.1103/PhysRevE.89.032703).

The **Region transforms** section in Mapping defines a mask relative to a supplied input, or from uploaded positions or intervals. It applies the same transforms as Data Manager to the initial fitted target within that mask, then holds those values fixed in every fitting step. Inputs remain unchanged. Regional constraints automatically select a compatible spatial or space–time target; time-resolved firing uses a free profile so the frozen regions can obey their own constraints. Constrained fork-speed fits use the existing deterministic mean-field solver with exact sensitivities and expose its control points, smoothness and convergence tolerance in Mapping settings. Frozen values override fit bounds locally and must remain valid kinetic values (non-negative initiation rates or positive fork speeds). Mask positions can be exported separately, and runnable Mapping downloads include auxiliary mask and modulation profiles.

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
