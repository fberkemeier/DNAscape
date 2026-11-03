"""Consistent, dismissible application messages."""

import re
from typing import Any

from nicegui import ui

_FILE_SUFFIX = (
    r"\.(?:csv|tsv|txt|npz|npy|npv|mat|bw|bigwig|bedgraph|bed|bdg|wig|"
    r"gz|zip|7z|xz|png|svg|pdf|json|html|py|xlsx|xls)"
    r"(?:\.(?:gz|zip|xz|7z))?"
)
_FILE_REFERENCE = re.compile(
    rf"""(?P<quote>['"`])(?:(?!(?P=quote))[^\r\n])+?{_FILE_SUFFIX}(?P=quote)
    |(?<!\w)(?:[A-Z]:[/\\]|/)[^\r\n"'`<>]*?{_FILE_SUFFIX}(?!\w)(?!\.[A-Za-z0-9])
    |(?<![\w.])(?:[A-Z]:)?[^\s"'`()\[\]{{}},;:<>]+{_FILE_SUFFIX}(?!\w)(?!\.[A-Za-z0-9])""",
    re.IGNORECASE | re.VERBOSE,
)


def _omit_file_reference(match: re.Match[str]) -> str:
    if re.search(r"\b(?:file|source|input|profile|track)\s+$", match.string[:match.start()], re.IGNORECASE):
        return ""
    return "the selected file"


def notification_message(message: Any) -> str:
    """Keep the reason, omitting file references from third-party errors too."""

    text = str(message).strip()
    text = re.sub(rf"^[^:\r\n]*{_FILE_SUFFIX}\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://[^\s<>\"'`]+", "the remote source", text)
    text = _FILE_REFERENCE.sub(_omit_file_reference, text)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    text = re.sub(r"^\[(?:Errno|WinError) \d+\]\s*", "", text)
    text = re.sub(r"^the selected file\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+(?:in|from) the selected file(?=[.;]?(?:$|\s))", "", text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r":\s*the selected file\.?$", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text[:1].upper() + text[1:]


def notify(message: Any, **options: Any) -> None:
    """Give every application message the same compact corner close button."""

    from dnascape._resources import LOCAL_SETUP_URL, hosted
    if hosted() and str(message).startswith(("Online memory limit:", "Online app busy:")):
        options["type"] = "warning"
        options["timeout"] = 0
        options["actions"] = [
            *(options.get("actions") or ()),
            {"label": "Run locally", "handler": lambda: ui.navigate.to(LOCAL_SETUP_URL, new_tab=True)},
        ]
    options["classes"] = (
        f"{options.get('classes') or ''} dn-notification dn-notification-dismissible"
    ).strip()
    if options.get("type") != "ongoing":
        options.setdefault("timeout", 3000)
    if "caption" in options:
        options["caption"] = notification_message(options["caption"])
    # Quasar's closeBtn=True creates a text label, not an icon. Actions
    # without a handler dismiss automatically when clicked.
    options["close_button"] = False
    options["actions"] = [
        *(options.get("actions") or ()),
        {
            "icon": "close", "round": True, "dense": True,
            "class": "dn-notification-close", "aria-label": "Dismiss message",
        },
    ]
    ui.notify(notification_message(message), **options)
