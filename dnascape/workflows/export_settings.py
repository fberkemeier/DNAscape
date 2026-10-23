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

DATA_FILE_FORMAT_OPTIONS = {
    "tsv": "TSV",
    "txt": "TXT",
    "csv": "CSV",
    "npz": "NPZ",
}

# Cache and workflow downloads deliberately share one public order.  Keep this
# separate object so callers cannot accidentally mutate the canonical mapping.
CACHE_FILE_FORMAT_OPTIONS = dict(DATA_FILE_FORMAT_OPTIONS)

DEFAULT_CONVERSION_MEMORY_LIMIT_MB = 64
MAX_CONVERSION_MEMORY_LIMIT_MB = 4096
DEFAULT_CONTROL_FONT_SIZE_PX = 14
MIN_CONTROL_FONT_SIZE_PX = 12
MAX_CONTROL_FONT_SIZE_PX = 18
DEFAULT_PREVIEW_POINTS = 1000
MIN_PREVIEW_POINTS = 100
MAX_PREVIEW_POINTS = 10000


@dataclass(slots=True)
class ExportSettings:
    """Mutable preferences shared by every workflow in the local app."""

    plot_dpi: int = 150
    raw_compression: str = "gzip"
    cached_file_format: str = "tsv"
    cached_compression: str = "gzip"
    result_name_style: str = "short"
    preserve_page_state: bool = True
    show_examples: bool = True
    show_help: bool = False
    dev_mode: bool = False
    conversion_memory_limit_mb: int = DEFAULT_CONVERSION_MEMORY_LIMIT_MB
    control_font_size_px: int = DEFAULT_CONTROL_FONT_SIZE_PX
    theme_color: str = "#b03412"
    preview_points: int = DEFAULT_PREVIEW_POINTS


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
    _EXPORT_SETTINGS.result_name_style = defaults.result_name_style
    _EXPORT_SETTINGS.preserve_page_state = defaults.preserve_page_state
    _EXPORT_SETTINGS.show_examples = defaults.show_examples
    _EXPORT_SETTINGS.show_help = defaults.show_help
    _EXPORT_SETTINGS.dev_mode = defaults.dev_mode
    _EXPORT_SETTINGS.conversion_memory_limit_mb = defaults.conversion_memory_limit_mb
    _EXPORT_SETTINGS.control_font_size_px = defaults.control_font_size_px
    _EXPORT_SETTINGS.theme_color = defaults.theme_color
    _EXPORT_SETTINGS.preview_points = defaults.preview_points
    return _EXPORT_SETTINGS


def preview_point_limit() -> int:
    """Bound the per-curve display budget independently of native data size."""
    return max(MIN_PREVIEW_POINTS, min(MAX_PREVIEW_POINTS, int(_EXPORT_SETTINGS.preview_points)))


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
    "DATA_FILE_FORMAT_OPTIONS",
    "DEFAULT_CONVERSION_MEMORY_LIMIT_MB",
    "DEFAULT_CONTROL_FONT_SIZE_PX",
    "DEFAULT_PREVIEW_POINTS",
    "MIN_PREVIEW_POINTS",
    "MAX_PREVIEW_POINTS",
    "MAX_CONVERSION_MEMORY_LIMIT_MB",
    "MAX_CONTROL_FONT_SIZE_PX",
    "MIN_CONTROL_FONT_SIZE_PX",
    "ExportSettings",
    "get_export_settings",
    "preview_point_limit",
    "normalize_compression",
    "reset_export_settings",
]
