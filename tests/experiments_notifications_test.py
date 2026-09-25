"""Intervention alerts survive transport failure without repeating delivered mail."""

import json
import subprocess
from email import policy
from email.parser import BytesParser
from unittest.mock import MagicMock

import pytest

from exact.experiments import notifications

INCIDENT = {"id": "incident-1", "run_ids": ["E10"], "reason": "Repeated selection failure"}
COMMAND = {
    "transport": "command",
    "command": ["/usr/local/bin/cluster-mail", "--stdin"],
    "recipient": "owner@example.org",
    "sender": "cluster@example.org",
    "timeout_seconds": 5,
}


def test_unconfigured_alert_is_persistent_and_can_be_delivered_later(tmp_path, monkeypatch):
    alert = notifications.notify_intervention(tmp_path, INCIDENT, "needs_user", "Choose a policy")
    assert alert["delivery"] == "not_configured"
    assert json.loads(open(alert["path"]).read())["summary"] == "Choose a policy"
    sent = []
    monkeypatch.setattr(
        notifications.subprocess, "run", lambda *args, **kwargs: sent.append(kwargs)
    )
    delivered = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=COMMAND
    )
    assert delivered["delivery"] == "sent" and len(sent) == 1
    assert delivered["created_at"] == alert["created_at"]


def test_delivery_deduplicates_incident_and_outcome(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(
        notifications.subprocess, "run", lambda *args, **kwargs: sent.append(kwargs)
    )
    for _ in range(3):
        receipt = notifications.notify_intervention(
            tmp_path, INCIDENT, "needs_user", "Choose a policy", config=COMMAND
        )
    assert receipt["attempts"] == 1 and len(sent) == 1
    notifications.notify_intervention(
        tmp_path, INCIDENT, "incident_attempt_limit", "Three failed repairs", config=COMMAND
    )
    assert len(sent) == 2


def test_command_receives_message_without_shell_interpolation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        notifications.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    text = "Needs decision; $(touch /tmp/never) `false`"
    notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", text, config=COMMAND, handoff="/local/HANDOFF.md"
    )
    args, kwargs = calls[0]
    assert args == (COMMAND["command"],)
    assert "shell" not in kwargs
    assert kwargs["timeout"] == 5
    message = BytesParser(policy=policy.default).parsebytes(kwargs["input"])
    assert message["To"] == "owner@example.org"
    assert text in message.get_content()
    assert "/local/HANDOFF.md" in message.get_content()


def test_failed_delivery_retries_without_logging_sensitive_errors(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["mail", "secret-password"], stderr="credential")

    monkeypatch.setattr(notifications.subprocess, "run", fail)
    first = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=COMMAND
    )
    assert first["delivery"] == "failed" and first["attempts"] == 1
    content = open(first["path"]).read()
    assert "secret-password" not in content and "credential" not in content
    monkeypatch.setattr(notifications.subprocess, "run", lambda *args, **kwargs: None)
    second = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=COMMAND
    )
    assert second["delivery"] == "sent" and second["attempts"] == 2


def test_filesystem_failure_does_not_crash_monitor(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(notifications, "_write", fail)
    result = notifications.notify_intervention(tmp_path, INCIDENT, "needs_user", "Choose a policy")
    assert result["delivery"] == "failed"
    assert result["error"] == "Notification persistence failed (OSError)"


@pytest.mark.parametrize("change", [{"timeout_seconds": 301}, {"command": ["relative-command"]}])
def test_invalid_delivery_configuration_is_visible_without_executing(tmp_path, monkeypatch, change):
    sent = []
    monkeypatch.setattr(
        notifications.subprocess, "run", lambda *args, **kwargs: sent.append(kwargs)
    )
    result = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=COMMAND | change
    )
    assert result["delivery"] == "failed" and sent == []


def test_smtp_requires_tls_and_reads_password_only_from_environment(tmp_path, monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.send_message.return_value = {}
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(notifications.smtplib, "SMTP", factory)
    monkeypatch.setenv("EXACT_TEST_MAIL_PASSWORD", "secret-password")
    config = {
        "transport": "smtp",
        "host": "smtp.example.org",
        "recipient": "owner@example.org",
        "sender": "cluster@example.org",
        "username": "cluster",
        "password_env": "EXACT_TEST_MAIL_PASSWORD",
    }
    result = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=config
    )
    assert result["delivery"] == "sent"
    assert client.method_calls[0][0] == "starttls"
    client.login.assert_called_once_with("cluster", "secret-password")
    assert "secret-password" not in open(result["path"]).read()
    assert factory.call_args.kwargs["timeout"] == 15


def test_sendmail_absence_is_visible_and_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(notifications.shutil, "which", lambda _: None)
    original = notifications.Path.is_file
    monkeypatch.setattr(
        notifications.Path,
        "is_file",
        lambda path: False if str(path) == "/usr/sbin/sendmail" else original(path),
    )
    config = {key: COMMAND[key] for key in ("recipient", "sender")}
    result = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=config
    )
    assert result["delivery"] == "failed"
    assert result["error"] == "Notification delivery failed (FileNotFoundError)"
