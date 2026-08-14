# DNAscape NiceGUI interface prototype

This folder contains a local, presentation-only NiceGUI prototype inspired by
the existing Streamlit application in `app_streamlit/`.

The prototype includes the full navigation shell, responsive layouts, light
and dark themes, page sections, scientific controls, placeholder previews,
tabs, expanders, and visible action buttons. Navigation and ordinary controls
are interactive. Scientific actions, exports, and cache mutations are
deliberately inert and the prototype does not import or call `dnascape`.

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
- `app/pages/` contains the seven presentation-only pages.
- `app/static/styles.css` adapts the Streamlit visual language to NiceGUI and
  includes desktop, tablet, and mobile layouts.
- `app_streamlit/` remains the complete Streamlit application.

Backend integration should be added only after the information architecture
and interaction design have been approved.
