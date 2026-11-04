"""NiceGUI workers always receive a detached snapshot of the initiating client."""

from nicegui import run as _run

from app.export_settings import run_with_settings, snapshot_export_settings


def io_bound(callback, *args, **kwargs):
    settings = snapshot_export_settings()
    return _run.io_bound(run_with_settings, settings, callback, args, kwargs)


def cpu_bound(callback, *args, **kwargs):
    settings = snapshot_export_settings()
    return _run.cpu_bound(run_with_settings, settings, callback, args, kwargs)
