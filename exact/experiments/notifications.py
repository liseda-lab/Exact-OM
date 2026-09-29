"""Persistent intervention alerts and optional bounded email delivery, using only stdlib."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import smtplib
import ssl
import subprocess
import time
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
    label = {
        "problem_detected": "Problem detected; automatic repair pending",
        "recovered": "Experiment recovered",
    }.get(alert["outcome"], "Action required: " + alert["outcome"])
    message["Subject"] = "[Exact-OM] " + label
    # Reuse Message-ID after an uncertain delivery; local dedup cannot guarantee exactly-once SMTP.
    message["Message-ID"] = f"<exact-om-{alert['id']}@supervisor.local>"
    message.set_content(
        f"Exact-OM: {label}.\n\n"
        f"Outcome: {alert['outcome']}\n"
        f"Runs: {', '.join(alert['run_ids']) or 'supervisor'}\n"
        f"Reason: {alert['reason']}\n\n"
        f"{alert['summary']}\n\n"
        f"Handoff: {alert['handoff'] or 'See supervisor status and health receipts.'}\n"
        f"Alert: {alert['path']}\n"
        "If a decision is required, reply in the Codex conversation; email replies are not monitored.\n"
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


def _attempt(alert: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    alert.update(delivery="sending", attempts=alert["attempts"] + 1, last_attempt_at=now)
    path = Path(alert["path"])
    _write(path, alert)
    try:
        _deliver(_message(alert, config), config)
    except Exception as exc:
        # Command exit 3 is the Gmail adapter's uncertain-send result. Its journal
        # must be inspected rather than risking a duplicate notification.
        ambiguous = isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 3
        alert.update(
            delivery="ambiguous" if ambiguous else "failed",
            error=f"Notification delivery failed ({type(exc).__name__})",
            next_attempt_epoch=time.time() + min(3600, 60 * 2 ** min(alert["attempts"] - 1, 6)),
        )
    else:
        alert.update(delivery="sent", delivered_at=now)
        alert.pop("error", None)
        alert.pop("next_attempt_epoch", None)
    alert["updated_at"] = now
    _write(path, alert)
    return alert


def notify_intervention(
    directory: str | Path,
    incident: Mapping[str, Any],
    outcome: str,
    summary: str,
    *,
    config: Mapping[str, Any] | None = None,
    handoff: str = "",
    defer: bool = False,
) -> dict[str, Any]:
    """Persist one immutable message per incident/outcome; the monitor queues only.

    Deferred delivery is drained independently, including after the incident has
    disappeared. Existing bodies stay unchanged for safe Gmail deduplication.
    Synchronous delivery remains available for explicit transport checks.
    """
    ident = hashlib.sha256(json.dumps([incident["id"], outcome]).encode()).hexdigest()[:24]
    path = Path(directory) / "alerts" / f"{ident}.json"
    now = datetime.now(timezone.utc).isoformat()
    configured = bool(config and config.get("enabled", True))
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
        "delivery": "pending" if configured else "not_configured",
        "attempts": 0,
    }
    try:
        if path.exists():
            alert = json.loads(path.read_text())
            if defer or alert.get("delivery") in {"sent", "ambiguous", "sending"}:
                return alert
        else:
            _write(path, alert)
        if configured and not defer:
            return _attempt(alert, config)
    except Exception as exc:
        alert.update(
            delivery="failed", error=f"Notification persistence failed ({type(exc).__name__})"
        )
    return alert


def flush_notifications(directory: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Drain the durable outbox with bounded backoff, independently of repairs.

    One delivery lock spans restarts. The Gmail adapter has an additional durable
    receipt journal, allowing recovery when its caller died after an actual send.
    """
    directory = Path(directory)
    counts: dict[str, int] = {}
    if not config or not config.get("enabled", True):
        return {"status": "not_configured", "counts": counts}
    with (directory / "notification-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "another_delivery_worker", "counts": counts}
        for path in sorted((directory / "alerts").glob("*.json")):
            try:
                alert = json.loads(path.read_text())
                delivery = alert.get("delivery")
                if delivery not in {"sent", "ambiguous"} and time.time() >= alert.get(
                    "next_attempt_epoch", 0
                ):
                    if delivery == "sending" and config.get("transport") != "command":
                        alert.update(delivery="ambiguous", error="Delivery interrupted; inspect transport")
                        _write(path, alert)
                    else:
                        alert = _attempt(alert, config)
                delivery = alert.get("delivery", "unknown")
                counts[delivery] = counts.get(delivery, 0) + 1
            except (OSError, ValueError, KeyError, TypeError):
                # One corrupt/unavailable alert cannot block the rest of the outbox.
                counts["unreadable"] = counts.get("unreadable", 0) + 1
    result = {"status": "checked", "counts": counts, "checked_at": datetime.now(timezone.utc).isoformat()}
    _write(directory / "notification-status.json", result)
    return result
