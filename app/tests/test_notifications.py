"""Notifications keep useful reasons without long filenames or paths."""

import pytest

from app.notifications import notification_message, notify


@pytest.mark.parametrize(("message", "expected"), (
    ("a-very-long-profile.tsv: The values must be finite.", "The values must be finite."),
    ("A long profile name.tsv: The values must be finite.", "The values must be finite."),
    ("'A long profile name.mat.gz' must contain one profile.", "The selected file must contain one profile."),
    ("No numeric column in profile.csv.", "No numeric column."),
    ("Column 'signal' not found in /tmp/session/source.tsv.", "Column 'signal' not found."),
    (r"Column 'signal' not found in C:\Users\A Long Name\source.tsv.", "Column 'signal' not found."),
    ("[Errno 2] No such file or directory: '/tmp/A long name/data.npz'", "No such file or directory"),
    ("[WinError 5] Access is denied: 'C:\\Users\\A Long Name\\input.npy'", "Access is denied"),
    ("The input 'a.long.file.name.bedGraph.gz' is empty.", "The input is empty."),
    ("The request to https://example.org/long-file.mat.gz timed out.", "The request to the remote source timed out."),
    ("Value 0.001 is below the 0.01 minimum.", "Value 0.001 is below the 0.01 minimum."),
    ("Column 'S1' contains an infinite value.", "Column 'S1' contains an infinite value."),
    ("Use NPZ instead.", "Use NPZ instead."),
))
def test_notification_message_keeps_the_reason(message: str, expected: str) -> None:
    assert notification_message(message) == expected
    assert notification_message(expected) == expected


@pytest.mark.parametrize("message_type", ("positive", "negative", "warning", "info", None))
def test_messages_have_shorter_timeout_and_retain_dismiss_actions(monkeypatch, message_type) -> None:
    calls = []
    monkeypatch.setattr("app.notifications.ui.notify", lambda message, **options: calls.append((message, options)))
    notify("profile.csv: The data is empty.", type=message_type, classes="custom", actions=[{"label": "Details"}])
    message, options = calls[0]
    assert message == "The data is empty."
    assert options["timeout"] == 3000
    assert options["type"] == message_type
    assert "custom" in options["classes"]
    assert "dn-notification-dismissible" in options["classes"]
    assert options["actions"][0] == {"label": "Details"}
    assert options["actions"][1]["icon"] == "close"
    assert options["actions"][1]["round"] is True
    assert options["close_button"] is False


def test_explicit_durations_and_ongoing_notifications_are_preserved(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("app.notifications.ui.notify", lambda message, **options: calls.append(options))
    notify("Check the data.", timeout=0)
    notify("Working...", type="ongoing")
    assert calls[0]["timeout"] == 0
    assert "timeout" not in calls[1]
