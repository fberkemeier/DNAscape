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


@dataclass(slots=True)
class ExportSettings:
    """Mutable preferences shared by every workflow in the local app."""

    plot_dpi: int = 150
    raw_compression: str = "gzip"


_EXPORT_SETTINGS = ExportSettings()


def get_export_settings() -> ExportSettings:
    """Return the process-local export preferences."""

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
    "COMPRESSION_OPTIONS",
    "ExportSettings",
    "get_export_settings",
    "normalize_compression",
]
