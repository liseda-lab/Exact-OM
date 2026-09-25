"""The Gmail adapter must verify actual tool receipts and never repeat an uncertain send."""

import importlib.util
import json
import subprocess
from email.message import EmailMessage
from pathlib import Path

import pytest


@pytest.fixture
def mailer(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/send_supervisor_email.py"
    spec = importlib.util.spec_from_file_location("supervisor_email_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "authenticate", lambda _: None)
    return module


def _mail(recipient="owner@example.org"):
    message = EmailMessage()
    message["To"] = recipient
    message["From"] = "cluster@example.org"
    message["Subject"] = "[Exact-OM] Intervention needed"
    message.set_content("Please decide the E01 ontology scope.")
    return message


def _policy():
    return {"codex": "/example/codex", "repository": "/example/repo"}


def _event(arguments, *, message_id="gmail-message-id", error=False):
    return {
        "type": "item.completed",
        "item": {
            "id": "tool-1",
            "type": "mcp_tool_call",
            "server": "codex_apps",
            "tool": "gmail_send_email",
            "arguments": arguments,
            "status": "completed",
            "result": {
                "isError": error,
                "content": [{"type": "text", "text": json.dumps({"id": message_id})}],
            },
        },
    }


def _child(mailer, monkeypatch, events, *, timeout=False):
    calls = []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, prompt, *, timeout):
            calls[-1][1]["prompt"] = prompt
            for event in events:
                self.kwargs["stdout"].write(json.dumps(event) + "\n")
            self.kwargs["stdout"].flush()
            if self.timeout:
                raise subprocess.TimeoutExpired("codex", timeout)

        def send_signal(self, signal):
            self.returncode = -signal

        def wait(self, timeout=None):
            return self.returncode

    Process.timeout = timeout
    monkeypatch.setattr(mailer.subprocess, "Popen", Process)
    return calls


def test_only_configured_plain_text_recipient_is_accepted(mailer):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    assert message["to"] == "owner@example.org"
    assert message["payload"]["body"]["content"] == "Please decide the E01 ontology scope.\n"
    with pytest.raises(ValueError):
        mailer.parse_message(_mail("other@example.org").as_bytes(), "owner@example.org")
    with pytest.raises(ValueError):
        mailer.parse_message(b"x" * 65537, "owner@example.org")


@pytest.mark.parametrize("header", ["Cc", "Bcc", "To"])
def test_added_recipients_are_rejected(mailer, header):
    message = _mail()
    # Bytes allow duplicate To although EmailMessage's writer rejects that header assignment.
    raw = (f"{header}: other@example.org\n").encode() + message.as_bytes()
    with pytest.raises(ValueError):
        mailer.parse_message(raw, "owner@example.org")


def test_success_requires_tool_receipt_and_is_deduplicated(mailer, tmp_path, monkeypatch):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    calls = _child(
        mailer,
        monkeypatch,
        [_event(message), {"type": "turn.completed", "usage": {"output_tokens": 15}}],
    )
    first = mailer.deliver(_policy(), tmp_path, message)
    second = mailer.deliver(_policy(), tmp_path, message)
    assert first["status"] == second["status"] == "sent" and len(calls) == 1
    assert first["receipts"] == [{"tool_call_id": "tool-1", "message_id": "gmail-message-id"}]
    assert first["usage"] == [{"output_tokens": 15}]
    assert first["message_sha256"] in first["directory"]
    assert "--approve-for-me" in calls[0][0] and "--ephemeral" in calls[0][0]
    assert 'forced_login_method="chatgpt"' in calls[0][0]
    assert "user explicitly authorized" in calls[0][1]["prompt"]
    assert json.loads((Path(first["directory"]) / "delivery.json").read_text())["status"] == "sent"


@pytest.mark.parametrize("evidence", ["prose", "wrong_recipient", "tool_error", "timeout"])
def test_uncertain_send_is_not_retried(mailer, tmp_path, monkeypatch, evidence):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    event = _event(message)
    if evidence == "prose":
        event = {"type": "item.completed", "item": {"type": "agent_message", "text": "Email sent!"}}
    elif evidence == "wrong_recipient":
        event["item"]["arguments"] = message | {"to": "other@example.org"}
    elif evidence == "tool_error":
        event["item"]["result"]["isError"] = True
    calls = _child(
        mailer, monkeypatch, [] if evidence == "timeout" else [event], timeout=evidence == "timeout"
    )
    first = mailer.deliver(_policy(), tmp_path, message)
    second = mailer.deliver(_policy(), tmp_path, message)
    assert first["status"] == second["status"] == "ambiguous" and len(calls) == 1


def test_configuration_failure_before_send_can_be_retried(mailer, tmp_path, monkeypatch):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    calls = _child(mailer, monkeypatch, [_event(message)])

    def failed(_):
        raise ValueError("private config error")

    monkeypatch.setattr(mailer, "authenticate", failed)
    first = mailer.deliver(_policy(), tmp_path, message)
    assert first["status"] == "failed" and calls == []
    assert "private config" not in first["error"]
    monkeypatch.setattr(mailer, "authenticate", lambda _: None)
    assert mailer.deliver(_policy(), tmp_path, message)["status"] == "sent"


def test_saved_success_receipt_recovers_interrupted_notifier(mailer, tmp_path, monkeypatch):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    calls = _child(mailer, monkeypatch, [_event(message)])
    first = mailer.deliver(_policy(), tmp_path, message)
    state_path = Path(first["directory"]) / "delivery.json"
    state = json.loads(state_path.read_text())
    state["status"] = "sending"
    state["error"] = "Stale uncertain-delivery warning"
    state_path.write_text(json.dumps(state))
    recovered = mailer.deliver(_policy(), tmp_path, message)
    assert recovered["status"] == "sent"
    assert "error" not in recovered
    assert len(calls) == 1


def test_changed_body_cannot_repeat_ambiguous_incident(mailer, tmp_path, monkeypatch):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    calls = _child(mailer, monkeypatch, [])
    first = mailer.deliver(
        _policy(), tmp_path, message, alert_id="<same-incident@supervisor.local>"
    )
    changed = message | {"subject": "[Exact-OM] Updated summary"}
    second = mailer.deliver(
        _policy(), tmp_path, changed, alert_id="<same-incident@supervisor.local>"
    )
    assert first["status"] == second["status"] == "ambiguous"
    assert first["directory"] == second["directory"] and len(calls) == 1


def test_real_cli_dotted_tool_name_and_structured_content_receipt(mailer, tmp_path):
    message = mailer.parse_message(_mail().as_bytes(), "owner@example.org")
    event = _event(message)
    event["item"]["tool"] = "gmail.send_email"
    event["item"]["result"] = {
        "content": [{"type": "text", "text": "Action completed."}],
        "structured_content": {"id": "1a0da2c7915d93b8", "thread_id": None, "label_ids": None},
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event) + "\n")
    assert mailer.delivery_evidence(path, message)["receipts"] == [
        {"tool_call_id": "tool-1", "message_id": "1a0da2c7915d93b8"}
    ]
