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


def test_pending_delivery_survives_incident_resolution_and_uses_backoff(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(notifications.time, "time", lambda: clock[0])
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(1)
        raise subprocess.CalledProcessError(2, ["mail"])
    monkeypatch.setattr(notifications.subprocess, "run", fail)
    alert = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Choose a policy", config=COMMAND, defer=True
    )
    assert alert["delivery"] == "pending" and not attempts
    notifications.flush_notifications(tmp_path, COMMAND)
    notifications.flush_notifications(tmp_path, COMMAND)
    assert len(attempts) == 1
    clock[0] += 61
    monkeypatch.setattr(notifications.subprocess, "run", lambda *a, **k: attempts.append(1))
    notifications.flush_notifications(tmp_path, COMMAND)
    assert json.loads(open(alert["path"]).read())["delivery"] == "sent"
    assert len(attempts) == 2


def test_ambiguous_delivery_does_not_retry_automatically(tmp_path, monkeypatch):
    def uncertain(*args, **kwargs):
        raise subprocess.CalledProcessError(3, ["mail"])
    monkeypatch.setattr(notifications.subprocess, "run", uncertain)
    alert = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Decide", config=COMMAND, defer=True
    )
    assert notifications.flush_notifications(tmp_path, COMMAND)["counts"] == {"ambiguous": 1}
    monkeypatch.setattr(notifications.subprocess, "run", lambda *a, **k: pytest.fail("Ambiguous send"))
    notifications.flush_notifications(tmp_path, COMMAND)
    assert json.loads(open(alert["path"]).read())["attempts"] == 1


def test_existing_message_is_immutable_while_queued(tmp_path):
    first = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Original", config=COMMAND, defer=True
    )
    second = notifications.notify_intervention(
        tmp_path, INCIDENT, "needs_user", "Changed", config=COMMAND, defer=True
    )
    assert first == second


def test_requested_token_milestone_delivers_once_without_action_required_label(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(notifications, "_deliver", lambda message, _: sent.append(message))
    for _ in range(3):
        notifications.notify_intervention(
            tmp_path, INCIDENT, "hosted_token_milestone", "100M tokens; experiments continue",
            config=COMMAND, defer=True,
        )
        notifications.flush_notifications(tmp_path, COMMAND)
    assert len(sent) == 1
    assert str(sent[0]["Subject"]) == "[Exact-OM] Hosted token milestone reached"
    assert "experiments continue" in sent[0].get_content()


@pytest.mark.parametrize("delivery", ["pending", "failed", "ambiguous"])
def test_superseded_user_decision_preserves_delivery_evidence_without_resend(
    tmp_path, monkeypatch, delivery
):
    alert = notifications.notify_intervention(
        tmp_path, INCIDENT, "requires_user", "Authorize larger spending",
        config=COMMAND, defer=True,
    )
    alert.update(delivery=delivery, attempts=1, error="original uncertainty", resolution={
        "kind": "superseded_by_user_authorization",
        "authorization": "User authorized continued spending on 2026-10-04",
        "resolved_at": "2026-10-04T18:00:00+00:00",
    })
    path = notifications.Path(alert["path"])
    notifications._write(path, alert)
    before = path.read_bytes()
    monkeypatch.setattr(notifications, "_deliver", lambda *args: pytest.fail("Obsolete decision"))
    assert notifications.notification_incidents(tmp_path) == []
    notifications.notify_intervention(
        tmp_path, INCIDENT, "requires_user", "Changed", config=COMMAND,
    )
    notifications.flush_notifications(tmp_path, COMMAND)
    assert path.read_bytes() == before


def test_incomplete_resolution_does_not_hide_delivery_uncertainty(tmp_path):
    alert = notifications.notify_intervention(tmp_path, INCIDENT, "requires_user", "Decide")
    alert.update(delivery="ambiguous", resolution={"kind": "superseded_by_user_authorization"})
    notifications._write(notifications.Path(alert["path"]), alert)
    assert len(notifications.notification_incidents(tmp_path)) == 1


def test_ambiguous_send_creates_one_repair_incident_without_recursive_alerts(tmp_path):
    alert = notifications.notify_intervention(
        tmp_path, INCIDENT, "requires_user", "Decide", config=COMMAND, defer=True
    )
    alert.update(delivery="ambiguous", needs_attention=True)
    notifications._write(notifications.Path(alert["path"]), alert)
    incident, = notifications.notification_incidents(tmp_path)
    assert incident["kind"] == "notification_delivery_uncertain"
    assert incident["alert_path"] == alert["path"]
    escalation = notifications.notify_intervention(
        tmp_path, incident, "requires_user", "Inspect delivery", config=COMMAND, defer=True
    )
    escalation.update(delivery="ambiguous")
    notifications._write(notifications.Path(escalation["path"]), escalation)
    assert notifications.notification_incidents(tmp_path) == [incident]
    alert.update(delivery="sent", needs_attention=False)
    notifications._write(notifications.Path(alert["path"]), alert)
    assert notifications.notification_incidents(tmp_path) == []


@pytest.mark.parametrize("outcome", [
    "problem_detected", "recovered", "repair_failed", "no_change", "daily_agent_limit",
    "supervisor_error", "unknown_information",
])
@pytest.mark.parametrize("defer", [False, True])
def test_only_action_required_outcomes_enter_delivery_queue(tmp_path, monkeypatch, outcome, defer):
    monkeypatch.setattr(notifications, "_deliver", lambda *args: pytest.fail("Informational email"))
    alert = notifications.notify_intervention(
        tmp_path, INCIDENT, outcome, "Automatic checks continue", config=COMMAND, defer=defer
    )
    assert alert["delivery"] == "suppressed" and alert["attempts"] == 0
    assert notifications.flush_notifications(tmp_path, COMMAND)["counts"] == {"suppressed": 1}


@pytest.mark.parametrize("outcome", [
    "needs_user", "requires_user", "approval_needed", "incident_attempt_limit",
    "supervisor_unavailable",
])
def test_action_required_outcomes_deliver_once(tmp_path, monkeypatch, outcome):
    sent = []
    monkeypatch.setattr(notifications, "_deliver", lambda *args: sent.append(args))
    notifications.notify_intervention(
        tmp_path, INCIDENT, outcome, "Human action required", config=COMMAND, defer=True
    )
    for _ in range(2):
        assert notifications.flush_notifications(tmp_path, COMMAND)["counts"] == {"sent": 1}
    assert len(sent) == 1


@pytest.mark.parametrize("delivery", ["pending", "not_configured", "failed", "sending", "sent", "ambiguous"])
@pytest.mark.parametrize("entrypoint", ["notify", "flush"])
def test_old_informational_alerts_never_send_and_preserve_delivery_evidence(
    tmp_path, monkeypatch, delivery, entrypoint
):
    monkeypatch.setattr(notifications, "_deliver", lambda *args: pytest.fail("Old informational email"))
    alert = notifications.notify_intervention(
        tmp_path, INCIDENT, "recovered", "Old immutable body", config=COMMAND, defer=True
    )
    alert.update(delivery=delivery, attempts=2, last_attempt_at="prior-attempt", next_attempt_epoch=1e20)
    if delivery == "sent":
        alert["delivered_at"] = "prior-delivery"
    if delivery == "ambiguous":
        alert["needs_attention"] = True
    notifications._write(notifications.Path(alert["path"]), alert)
    before = notifications.Path(alert["path"]).read_bytes()
    if entrypoint == "notify":
        notifications.notify_intervention(
            tmp_path, INCIDENT, "recovered", "Changed body", config=COMMAND
        )
    else:
        notifications.flush_notifications(tmp_path, COMMAND)
    after = json.loads(notifications.Path(alert["path"]).read_text())
    assert after["summary"] == "Old immutable body"
    assert after["attempts"] == 2 and after["last_attempt_at"] == "prior-attempt"
    if delivery in {"sent", "ambiguous"}:
        assert notifications.Path(alert["path"]).read_bytes() == before
    elif delivery == "sending":
        assert after["delivery"] == "ambiguous" and after["needs_attention"]
        assert len(notifications.notification_incidents(tmp_path)) == 1
    else:
        assert after["delivery"] == "suppressed" and after["suppressed_delivery"] == delivery
