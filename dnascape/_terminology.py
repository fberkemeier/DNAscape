"""Canonical initiation-rate names at legacy input boundaries."""

from collections.abc import Mapping


def canonical_rate_key(value):
    text = str(value)
    if text in {"ir", "fr"}:
        return "initiation_rate"
    return text.replace("firing_rate", "initiation_rate")


def canonical_recipe(value, field=None):
    """Update semantic keys while preserving paths, columns and user filenames."""
    if isinstance(value, Mapping):
        return {canonical_rate_key(key): canonical_recipe(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical_recipe(item, field) for item in value]
    if isinstance(value, str) and field in {"method", "target", "output", "relative_to"}:
        return canonical_rate_key(value)
    return value
