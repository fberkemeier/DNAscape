"""Application-wide export preferences for the local NiceGUI interface."""

from __future__ import annotations

from dataclasses import dataclass

COMPRESSION_OPTIONS = {
    "gzip": "GZIP (.gz)",
    "zip": "ZIP (.zip)",
    "7z": "7-Zip (.7z)",
    "xz": "XZ (.xz)",
    "none": "None",
}

CACHE_FILE_FORMAT_OPTIONS = {
    "tsv": "TSV",
    "csv": "CSV",
    "npz": "NPZ",
    "npy": "NPY",
    "npv": "NPV",
    "txt": "TXT",
}


@dataclass(slots=True)
class ExportSettings:
    """Mutable preferences shared by every workflow in the local app."""

    plot_dpi: int = 150
    raw_compression: str = "gzip"
    cached_file_format: str = "tsv"
    cached_compression: str = "gzip"
    preserve_page_state: bool = True


_EXPORT_SETTINGS = ExportSettings()


def get_export_settings() -> ExportSettings:
    """Return the process-local export preferences."""

    return _EXPORT_SETTINGS


def reset_export_settings() -> ExportSettings:
    """Restore every application-wide export preference to its default."""

    defaults = ExportSettings()
    _EXPORT_SETTINGS.plot_dpi = defaults.plot_dpi
    _EXPORT_SETTINGS.raw_compression = defaults.raw_compression
    _EXPORT_SETTINGS.cached_file_format = defaults.cached_file_format
    _EXPORT_SETTINGS.cached_compression = defaults.cached_compression
    _EXPORT_SETTINGS.preserve_page_state = defaults.preserve_page_state
    return _EXPORT_SETTINGS


def normalize_compression(value: str | None) -> str:
    """Normalize one menu value to a supported archive format."""

    normalized = str(value or "none").strip().casefold().lstrip(".")
    aliases = {
        "gz": "gzip",
        "gzip": "gzip",
        "zip": "zip",
        "7zip": "7z",
        "7-zip": "7z",
        "7z": "7z",
        "lzma": "xz",
        "xz": "xz",
        "none": "none",
        "off": "none",
        "": "none",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(COMPRESSION_OPTIONS)
        raise ValueError(f"Unsupported compression {value!r}; choose {choices}.") from exc


__all__ = [
    "CACHE_FILE_FORMAT_OPTIONS",
    "COMPRESSION_OPTIONS",
    "ExportSettings",
    "get_export_settings",
    "normalize_compression",
    "reset_export_settings",
]
