# DNAscape NiceGUI interface prototype

This folder contains the local DNAscape NiceGUI interface.

The prototype includes the full navigation shell, responsive layouts, light
and dark themes, page sections, scientific controls, placeholder previews,
tabs, expanders, and visible action buttons. Navigation and ordinary controls
are interactive. The Simulation page now provides the first live scientific
path: constant firing-rate and fork-speed inputs run through `dnascape.rsim`
in a background worker, with progress and a result plot in Preview. Plot
downloads support PNG, SVG, and PDF; selected raw outputs support text, NPY,
and NPZ downloads with optional gzip. Detailed time-series and space-time
statistics are available behind an off-by-default switch so ordinary runs stay
fast and memory-bounded. Profile inputs and the remaining scientific actions
and cache mutations are still being connected incrementally.

## Install and launch

From the repository root:

The prototype requires Python 3.10 or newer (the scientific package itself
continues to support its existing Python range).

```powershell
python -m pip install -r app/requirements.txt
python -m app
```

The local interface opens at `http://127.0.0.1:8080`. To choose another port:

```powershell
$env:DNASCAPE_APP_PORT = "8090"
python -m app
```

For automated or headless launch checks, set `DNASCAPE_APP_SHOW=false`.

The logos are served directly from `docs/source/_static/`; they are not copied
or duplicated in this folder.

## Prototype boundary

- `app/main.py` owns the NiceGUI process and persistent shell.
- `app/pages/` contains the seven workflow pages.
- `app/simulation_service.py` validates bounded constant-input jobs and bridges
  the Simulation page to `dnascape.rsim` without coupling scientific code to
  NiceGUI elements.
- `app/simulation_exports.py` lazily creates bounded plot and raw-data downloads
  from completed Simulation results.
- `app/static/styles.css` implements the DNAscape visual language and includes
  desktop, tablet, and mobile layouts.

The constant-input Simulation path is the first backend integration. Mapping,
profile uploads, and session persistence remain subsequent steps.
