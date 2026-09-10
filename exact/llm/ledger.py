"""Durable OpenRouter requests and attempts, independent of response parsing.

Every wire attempt has its own row. Unknown delivery remains chargeable; completed
responses replay without another paid request. SQLite serializes concurrent claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def request_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Canonical identity contains exact public request semantics, never credentials."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


class RequestLedger:
    """Persist raw response bytes before parsing and serialize each request's sender."""

    def __init__(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "requests.sqlite3"
        with self._transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY, identity TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    request_id TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
                    pid INTEGER NOT NULL, host TEXT NOT NULL, status INTEGER,
                    raw BLOB, sha256 TEXT, usage TEXT, error TEXT,
                    started TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(request_id, number)
                );
            """
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def plan(self, identity: Mapping[str, Any]) -> str:
        """Durably record the exact planned request before any network activity."""
        key, encoded = request_identity(identity)
        with self._transaction() as db:
            db.execute("INSERT OR IGNORE INTO requests VALUES (?, ?)", (key, encoded))
        return key

    def cached(self, key: str) -> bytes | None:
        """Return verified raw bytes from the latest completed wire attempt."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT raw, sha256 FROM attempts WHERE request_id=? "
                "AND state='completed' ORDER BY number DESC LIMIT 1",
                (key,),
            ).fetchone()
        if row is None:
            return None
        raw = bytes(row["raw"])
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Corrupted completed OpenRouter response {key}")
        return raw

    def sent(self, key: str, *, retry_unknown: bool = False) -> int:
        """Claim one wire attempt; duplicate active writers and unapproved retries fail."""
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM requests WHERE request_id=?", (key,)).fetchone() is None:
                raise ValueError("Request was not planned")
            last = db.execute(
                "SELECT * FROM attempts WHERE request_id=? ORDER BY number DESC LIMIT 1", (key,)
            ).fetchone()
            if last is not None:
                if last["state"] == "completed":
                    raise RuntimeError("Request completed concurrently; reload its cached response")
                if last["state"] == "sent":
                    raise RuntimeError(
                        "Request has an active or unresolved sender; recover it explicitly"
                    )
                if last["state"] == "unknown" and not retry_unknown:
                    raise RuntimeError(
                        "Unknown paid request; explicit retry authorization is required"
                    )
            number = int(last["number"]) + 1 if last else 1
            db.execute(
                "INSERT INTO attempts(request_id,number,state,pid,host) VALUES (?,?,?,?,?)",
                (key, number, "sent", os.getpid(), socket.gethostname()),
            )
        return number

    def received(self, key: str, number: int, raw: bytes, status: int) -> None:
        """Commit exact response bytes before decoding/parsing; errors remain retry records."""
        state = "completed" if 200 <= status < 300 else "rejected"
        with self._transaction() as db:
            changed = db.execute(
                "UPDATE attempts SET state=?,raw=?,sha256=?,status=? "
                "WHERE request_id=? AND number=? AND state='sent'",
                (state, raw, hashlib.sha256(raw).hexdigest(), status, key, number),
            )
            if changed.rowcount != 1:
                raise ValueError("Wire attempt is not pending")

    def usage(self, key: str, number: int, usage: Mapping[str, Any]) -> None:
        """Record provider-reported actual usage without inventing missing prices."""
        with self._transaction() as db:
            db.execute(
                "UPDATE attempts SET usage=? WHERE request_id=? AND number=?",
                (json.dumps(dict(usage), sort_keys=True), key, number),
            )

    def unknown(self, key: str, number: int, error: str) -> None:
        """Mark ambiguous transport failure; retries retain this possibly charged attempt."""
        with self._transaction() as db:
            db.execute(
                "UPDATE attempts SET state='unknown',error=? "
                "WHERE request_id=? AND number=? AND state='sent'",
                (error, key, number),
            )

    def recover_unknown(self, key: str) -> None:
        """Explicitly recover a crashed local sender only after proving its process exited."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE request_id=? AND state='sent' "
                "ORDER BY number DESC LIMIT 1",
                (key,),
            ).fetchone()
            if row is None:
                return
            if row["host"] != socket.gethostname():
                raise ValueError("Cannot establish that a remote sender exited")
            try:
                os.kill(int(row["pid"]), 0)
            except ProcessLookupError:
                db.execute(
                    "UPDATE attempts SET state='unknown',error='sender exited' "
                    "WHERE request_id=? AND number=?",
                    (key, row["number"]),
                )
            else:
                raise ValueError("Sender is still alive")

    def summary(self) -> dict[str, Any]:
        """Report all attempts, actual usage, and explicitly unresolved cost exposure."""
        with self._transaction() as db:
            rows = db.execute(
                "SELECT a.*,r.identity FROM attempts a JOIN requests r USING(request_id)"
            ).fetchall()
        roles: dict[str, Any] = {}
        for row in rows:
            role = json.loads(row["identity"]).get("role", "unspecified")
            totals = roles.setdefault(
                role,
                {
                    "attempts": 0,
                    "completed": 0,
                    "unknown": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reported_cost_usd": 0.0,
                    "unpriced_attempts": 0,
                },
            )
            totals["attempts"] += 1
            if row["state"] == "completed":
                totals["completed"] += 1
            if row["state"] in {"unknown", "sent"}:
                totals["unknown"] += 1
            usage = json.loads(row["usage"] or "{}")
            totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            if usage.get("cost") is None:
                totals["unpriced_attempts"] += 1
            else:
                totals["reported_cost_usd"] += float(usage["cost"])
        return {"roles": roles, "exactly_once_guaranteed": False}
