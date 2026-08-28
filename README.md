<p align="center">
  <img src="docs/source/_static/dnascape_logo.svg" alt="DNAscape" width="420">
</p>

# DNAscape

DNAscape is a local browser app for preparing, mapping, simulating, and plotting DNA replication data. It brings the main DNAscape workflows into one interface while keeping the computation on your computer.

## What you can do

- **Prepare data:** import tables, NumPy profiles, bedGraph, or BigWig tracks; select genomic regions; bin, smooth, rescale, invert, or filter values; and export DNAscape-ready profiles.
- **Map replication features:** infer supported kinetic targets from one or more compatible input profiles or constant values.
- **Run simulations:** simulate replication from firing-rate and fork-speed profiles or constants, compare datasets, and inspect standard or detailed outputs.
- **Plot results:** load multiple profiles, view them together or separately, adjust their presentation, and export publication-ready plots.
- **Reuse recent work:** reopen or download files and completed workflows retained in the current browser session.

The app accepts common profile formats including CSV, TSV, TXT, NPY, NPZ, bedGraph, and BigWig. The Data Manager can also import public BigWig or bedGraph links.

## Run locally

### Requirements

- Python 3.10 or newer
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

## Local data and sessions

DNAscape does not require a hosted service. Uploaded files and calculations are handled by the Python process running on your computer. A public URL is accessed only when you explicitly choose it as a data source.

Recent uploads and results are cached in memory for the current browser tab and are not permanent storage. Use **Cached Files** to manage them, or **Reset to defaults** in the top-right menu to clear the workspace.

## Troubleshooting

- **The page did not open automatically:** visit [http://127.0.0.1:8080](http://127.0.0.1:8080) manually.
- **Port 8080 is already in use:** set `DNASCAPE_APP_PORT` to another port before running `python -m app`.
- **A change is not visible:** stop the app with `Ctrl+C`, start it again, and refresh the browser.
- **PowerShell blocks environment activation:** skip activation and run `.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt`, followed by `.\.venv\Scripts\python.exe -m app`.

## Project status

The local interface is under active development. Please report bugs or suggest improvements through [GitHub Issues](https://github.com/fberkemeier/DNAscape/issues).

DNAscape is available under the [MIT License](LICENSE).
