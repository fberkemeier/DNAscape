"""Framework-independent preferences and explicit per-job settings scopes."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, replace

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
    """Personal preferences; browser clients own separate instances."""

    plot_dpi: int = 150
    raw_compression: str = "gzip"
    cached_file_format: str = "tsv"
    cached_compression: str = "gzip"
    result_name_style: str = "short"
    preserve_page_state: bool = True
    show_examples: bool = True
    show_field_hints: bool = True
    show_help: bool = False
    dev_mode: bool = False
    conversion_memory_limit_mb: int = DEFAULT_CONVERSION_MEMORY_LIMIT_MB
    control_font_size_px: int = DEFAULT_CONTROL_FONT_SIZE_PX
    theme_color: str = "#b03412"
    preview_points: int = DEFAULT_PREVIEW_POINTS
    theme_mode: str = "system"


_HEADLESS_SETTINGS = ExportSettings()
_SETTINGS_SCOPE: ContextVar[ExportSettings | None] = ContextVar("dnascape_settings", default=None)
_settings_provider: Callable[[], ExportSettings | None] | None = None


def set_settings_provider(provider: Callable[[], ExportSettings | None]) -> None:
    """Let a front end resolve its client without importing that framework here."""
    global _settings_provider
    _settings_provider = provider


def get_export_settings() -> ExportSettings:
    """Resolve a job snapshot, a front-end client, or standalone Python defaults."""
    scoped = _SETTINGS_SCOPE.get()
    if scoped is not None:
        return scoped
    if _settings_provider is not None:
        current = _settings_provider()
        if current is not None:
            return current
    return _HEADLESS_SETTINGS


def snapshot_export_settings() -> ExportSettings:
    return replace(get_export_settings())


@contextmanager
def settings_scope(settings: ExportSettings):
    """Bind an independent copy for one calculation, thread or recipe."""
    token = _SETTINGS_SCOPE.set(replace(settings))
    try:
        yield get_export_settings()
    finally:
        _SETTINGS_SCOPE.reset(token)


def run_with_settings(settings, callback, args, kwargs):
    """Picklable worker entry point; no UI context crosses the boundary."""
    with settings_scope(settings):
        return callback(*args, **kwargs)


def load_export_settings(payload) -> ExportSettings:
    """Validate untrusted, versioned browser values, falling back field by field."""
    result = ExportSettings()
    if not isinstance(payload, Mapping):
        return result
    bounded = {"plot_dpi": (72, 384), "conversion_memory_limit_mb": (64, 4096),
               "control_font_size_px": (12, 18), "preview_points": (100, 10000)}
    choices = {"raw_compression": COMPRESSION_OPTIONS, "cached_compression": COMPRESSION_OPTIONS,
               "cached_file_format": CACHE_FILE_FORMAT_OPTIONS,
               "result_name_style": ("short", "full"), "theme_mode": ("system", "light", "dark")}
    for item in fields(result):
        value = payload.get(item.name)
        default = getattr(result, item.name)
        if item.name in bounded:
            low, high = bounded[item.name]
            if type(value) in (int, float) and low <= value <= high and float(value).is_integer():
                setattr(result, item.name, int(value))
        elif item.name in choices:
            if isinstance(value, str) and value in choices[item.name]:
                setattr(result, item.name, value)
        elif item.name == "theme_color":
            if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                result.theme_color = value.lower()
        elif isinstance(default, bool) and isinstance(value, bool):
            setattr(result, item.name, value)
    return result


def reset_export_settings() -> ExportSettings:
    """Reset only the current scope, keeping existing UI bindings intact."""
    settings = get_export_settings()
    for key, value in asdict(ExportSettings()).items():
        setattr(settings, key, value)
    return settings


def preview_point_limit() -> int:
    """Bound the per-curve display budget independently of native data size."""
    return max(MIN_PREVIEW_POINTS, min(MAX_PREVIEW_POINTS, int(get_export_settings().preview_points)))


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
    "snapshot_export_settings",
    "settings_scope",
    "run_with_settings",
    "load_export_settings",
    "set_settings_provider",
]
