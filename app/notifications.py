"""Consistent, dismissible application messages."""

from typing import Any

from nicegui import ui


def notify(message: Any, **options: Any) -> None:
    """Keep success and error messages dismissible with a compact close icon."""

    if options.get("type") in {"positive", "negative"}:
        options["close_button"] = True
        options["classes"] = f"{options.get('classes') or ''} dn-notification-dismissible".strip()
    ui.notify(message, **options)
