"""Persistent intervention alerts and optional bounded email delivery, using only stdlib."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import smtplib
import ssl
import subprocess
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _message(alert: Mapping[str, Any], config: Mapping[str, Any]) -> EmailMessage:
    recipient, sender = config.get("recipient"), config.get("sender")
    if not all(isinstance(value, str) and value.strip() for value in (recipient, sender)):
        raise ValueError("Notification sender and recipient must be configured")
    message = EmailMessage()
    message["To"], message["From"] = recipient, sender
    message["Subject"] = f"[Exact-OM] Intervention needed: {alert['outcome']}"
    # Reuse Message-ID after an uncertain delivery; local dedup cannot guarantee exactly-once SMTP.
    message["Message-ID"] = f"<exact-om-{alert['id']}@supervisor.local>"
    message.set_content(
        "Exact-OM needs your intervention.\n\n"
        f"Outcome: {alert['outcome']}\n"
        f"Runs: {', '.join(alert['run_ids']) or 'supervisor'}\n"
        f"Reason: {alert['reason']}\n\n"
        f"{alert['summary']}\n\n"
        f"Handoff: {alert['handoff'] or 'See supervisor status and health receipts.'}\n"
        f"Alert: {alert['path']}\n"
        "Give the required decision in the Codex conversation; email replies are not monitored.\n"
        "Other eligible experiments remain under supervision.\n"
    )
    return message


def _deliver(message: EmailMessage, config: Mapping[str, Any]) -> None:
    transport = config.get("transport", "sendmail")
    timeout = float(config.get("timeout_seconds", 15))
    maximum = 300 if transport == "command" else 30
    if not 0 < timeout <= maximum:
        raise ValueError(f"Notification timeout must be positive and at most {maximum} seconds")
    if transport in {"sendmail", "command"}:
        command = config.get("command")
        if command is None and transport == "sendmail":
            executable = shutil.which("sendmail")
            if not executable and Path("/usr/sbin/sendmail").is_file():
                executable = "/usr/sbin/sendmail"
            if not executable:
                raise FileNotFoundError("sendmail is unavailable")
            command = [executable, "-t", "-oi"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(arg, str) and "\x00" not in arg for arg in command)
            or not Path(command[0]).is_absolute()
        ):
            raise ValueError(
                "Notification command must be an argv list with an absolute executable"
            )
        # The reviewed argv is configuration, never interpolated from incident text.
        subprocess.run(
            command,
            input=message.as_bytes(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )
    elif transport == "smtp":
        host = config["host"]
        tls = config.get("tls", "starttls")
        if tls not in {"ssl", "starttls"}:
            raise ValueError("SMTP requires SSL or STARTTLS")
        context = ssl.create_default_context()
        port = int(config.get("port", 465 if tls == "ssl" else 587))
        connection = (
            smtplib.SMTP_SSL(host, port, timeout=timeout, context=context)
            if tls == "ssl"
            else smtplib.SMTP(host, port, timeout=timeout)
        )
        with connection as client:
            if tls == "starttls":
                client.starttls(context=context)
            username = config.get("username")
            if username:
                # Secrets are provided through the supervisor environment, never stored in policy.
                client.login(username, os.environ[config["password_env"]])
            refused = client.send_message(message)
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)
    else:
        raise ValueError("Unknown notification transport")


def notify_intervention(
    directory: str | Path,
    incident: Mapping[str, Any],
    outcome: str,
    summary: str,
    *,
    config: Mapping[str, Any] | None = None,
    handoff: str = "",
) -> dict[str, Any]:
    """Persist an alert, and send once per incident/outcome; failed deliveries retry.

    Call only for an intervention outcome, under the supervisor's existing lock.
    ``config`` supports recipient, sender and transport (sendmail, command or smtp).
    Command transport receives an RFC 822 message on stdin and uses a reviewed argv
    list, without a shell. SMTP passwords come only from ``password_env``. An absent
    config keeps a visible local alert that can be delivered after configuration.
    Returned delivery status must be surfaced in the supervisor status receipt.
    """
    ident = hashlib.sha256(json.dumps([incident["id"], outcome]).encode()).hexdigest()[:24]
    path = Path(directory) / "alerts" / f"{ident}.json"
    now = datetime.now(timezone.utc).isoformat()
    alert: dict[str, Any] = {
        "id": ident,
        "incident_id": incident["id"],
        "outcome": outcome,
        "run_ids": incident.get("run_ids", []),
        "reason": incident.get("reason", ""),
        "summary": summary,
        "handoff": handoff,
        "path": str(path),
        "created_at": now,
        "updated_at": now,
        "delivery": "not_configured",
        "attempts": 0,
    }
    try:
        if path.exists():
            previous: dict[str, Any] = json.loads(path.read_text())
            alert.update(
                {key: previous[key] for key in ("created_at", "attempts") if key in previous}
            )
            if previous.get("delivery") == "sent":
                return previous
        _write(path, alert)  # Preserve the actionable message even if delivery fails.
        if not config or not config.get("enabled", True):
            return alert
        alert.update(attempts=alert["attempts"] + 1, last_attempt_at=now)
        try:
            _deliver(_message(alert, config), config)
        except Exception as exc:
            # SMTP/command exceptions can contain credentials, response bodies or argv.
            alert.update(
                delivery="failed", error=f"Notification delivery failed ({type(exc).__name__})"
            )
        else:
            alert.update(delivery="sent", delivered_at=now)
        _write(path, alert)
    except Exception as exc:
        # Notification filesystem trouble must not stop repair or monitoring.
        alert.update(
            delivery="failed", error=f"Notification persistence failed ({type(exc).__name__})"
        )
    return alert
