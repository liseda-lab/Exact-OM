"""Durable OpenRouter requests and attempts, independent of response parsing.

Every wire attempt has its own row. Unknown delivery remains chargeable; completed
responses replay without another paid request. SQLite serializes concurrent claims.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import socket
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from exact.utils.hosted_spending import load_spending_policy


def request_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Canonical identity contains exact public request semantics, never credentials."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


def _validate_elapsed(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError("Elapsed wire time must be finite and nonnegative")


class RequestLedger:
    """Persist raw response bytes before parsing and serialize each request's sender."""

    def __init__(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "requests.sqlite3"
        with self._transaction() as db:
            # Individual statements retain the transaction lock during schema migration.
            for statement in (
                """CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY, identity TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS reservations (
                    request_id TEXT NOT NULL, number INTEGER NOT NULL, tokens INTEGER NOT NULL,
                    PRIMARY KEY(request_id, number)
                )""",
                """CREATE TABLE IF NOT EXISTS attempts (
                    request_id TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
                    pid INTEGER NOT NULL, host TEXT NOT NULL, status INTEGER,
                    raw BLOB, sha256 TEXT, usage TEXT, error TEXT,
                    started TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    elapsed_seconds REAL,
                    PRIMARY KEY(request_id, number)
                )""",
                """CREATE TABLE IF NOT EXISTS retry_authorizations (
                    request_id TEXT NOT NULL, number INTEGER NOT NULL,
                    authorization_id TEXT NOT NULL, consumed_by INTEGER,
                    PRIMARY KEY(request_id, number)
                )""",
                """CREATE TABLE IF NOT EXISTS spending_policies (
                    sha256 TEXT PRIMARY KEY, record TEXT NOT NULL
                )""",
                # The ledger is copied between frozen workers, including older
                # clients that do not know this table. Enforce consumed approvals
                # in SQLite too, so rejected retries cannot escape their limit.
                """CREATE TRIGGER IF NOT EXISTS enforce_one_shot_retry
                BEFORE INSERT ON attempts
                WHEN EXISTS (
                    SELECT 1 FROM retry_authorizations g
                    WHERE g.request_id=NEW.request_id AND g.consumed_by IS NOT NULL
                    AND NEW.number > g.consumed_by AND g.number=(
                        SELECT MAX(number) FROM retry_authorizations
                        WHERE request_id=NEW.request_id
                    )
                ) BEGIN
                    SELECT RAISE(ABORT, 'Approved paid retry already used');
                END""",
            ):
                db.execute(statement)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(attempts)")}
            if "elapsed_seconds" not in columns:
                # Prior attempts have no measured duration; never infer it from timestamps.
                db.execute("ALTER TABLE attempts ADD COLUMN elapsed_seconds REAL")
            if "spending_policy_sha256" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN spending_policy_sha256 TEXT")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        # Queue local writers before SQLite's bounded busy timeout, especially on
        # NFS. Never hold this lock during a hosted request or response parsing.
        with self.path.with_suffix(".transaction.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
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
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

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

    @contextmanager
    def request_lock(self, key: str) -> Iterator[None]:
        """Serialize an exact request through response commit, across local clients.

        Waiting callers must recheck the cache under this lock. A crashed sender
        releases the file lock but retains its sent/unknown row and retry guard.
        Distinct requests use separate locks and retain transport concurrency.
        """
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Invalid canonical request identity")
        directory = self.path.parent / "request-locks"
        directory.mkdir(exist_ok=True)
        with (directory / (key + ".lock")).open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def sent(self, key: str, *, retry_unknown: bool = False) -> int:
        """Claim one wire attempt; duplicate active writers and unapproved retries fail."""
        policy = load_spending_policy()
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM requests WHERE request_id=?", (key,)).fetchone() is None:
                raise ValueError("Request was not planned")
            last = db.execute(
                "SELECT * FROM attempts WHERE request_id=? ORDER BY number DESC LIMIT 1", (key,)
            ).fetchone()
            grant = db.execute(
                "SELECT * FROM retry_authorizations WHERE request_id=? "
                "ORDER BY number DESC LIMIT 1",
                (key,),
            ).fetchone()
            if last is not None:
                if last["state"] == "completed":
                    raise RuntimeError("Request completed concurrently; reload its cached response")
                if grant is not None and grant["consumed_by"] is not None:
                    raise RuntimeError(
                        "Approved paid retry already used; another attempt requires authorization"
                    )
                if last["state"] == "sent":
                    raise RuntimeError(
                        "Request has an active or unresolved sender; recover it explicitly"
                    )
                authorized = grant is not None and grant["number"] == last["number"]
                if last["state"] == "unknown" and not (retry_unknown or authorized):
                    raise RuntimeError(
                        "Unknown paid request; explicit retry authorization is required"
                    )
            number = int(last["number"]) + 1 if last else 1
            request_cap = os.getenv("EXACT_OPENROUTER_REQUEST_CAP")
            token_cap = os.getenv("EXACT_OPENROUTER_TOKEN_CAP")
            if request_cap or token_cap or policy:
                rows = db.execute(
                    "SELECT a.usage,r.tokens FROM attempts a LEFT JOIN reservations r USING(request_id,number)"
                ).fetchall()
                if request_cap and len(rows) + 1 > int(request_cap):
                    raise ValueError("OpenRouter request budget exhausted before transmission")
                identity = json.loads(
                    db.execute(
                        "SELECT identity FROM requests WHERE request_id=?", (key,)
                    ).fetchone()[0]
                )
                payload = identity.get("payload", {})
                if type(payload.get("max_tokens")) is not int or payload["max_tokens"] <= 0:
                    raise ValueError("Budgeted hosted requests need a finite max_tokens bound")
                # UTF-8 bytes bound text token counts conservatively; include framing overhead.
                reserved = (
                    len(json.dumps(payload, ensure_ascii=False).encode())
                    + 256
                    + int(payload["max_tokens"])
                )
                used = 0
                for row in rows:
                    usage = json.loads(row["usage"] or "{}")
                    if all(
                        usage.get(field) is not None
                        for field in ("prompt_tokens", "completion_tokens")
                    ):
                        used += int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
                    elif row["tokens"] is not None:
                        used += int(row["tokens"])
                    else:
                        raise ValueError("Existing unpriced request lacks a retained token bound")
                if token_cap and used + reserved > int(token_cap):
                    raise ValueError("OpenRouter token budget exhausted before transmission")
                db.execute("INSERT INTO reservations VALUES (?,?,?)", (key, number, reserved))
            if policy:
                record = json.dumps(policy, sort_keys=True, allow_nan=False)
                retained = db.execute(
                    "SELECT record FROM spending_policies WHERE sha256=?", (policy["sha256"],)
                ).fetchone()
                if retained is not None and retained["record"] != record:
                    raise ValueError("Retained hosted spending policy receipt changed")
                db.execute(
                    "INSERT OR IGNORE INTO spending_policies VALUES (?,?)",
                    (policy["sha256"], record),
                )
            db.execute(
                "INSERT INTO attempts(request_id,number,state,pid,host,spending_policy_sha256) "
                "VALUES (?,?,?,?,?,?)",
                (
                    key,
                    number,
                    "sent",
                    os.getpid(),
                    socket.gethostname(),
                    policy["sha256"] if policy else None,
                ),
            )
            if grant is not None:
                db.execute(
                    "UPDATE retry_authorizations SET consumed_by=? "
                    "WHERE request_id=? AND number=? AND consumed_by IS NULL",
                    (number, key, grant["number"]),
                )
        return number

    def authorize_unknown_once(self, key: str, *, attempt: int, authorization_id: str) -> None:
        """Bind explicit approval to one extra wire attempt, retaining uncertain charges.

        Reapplying the same approval is idempotent, including after it was used.
        Approval and dead-sender recovery share a transaction; no network is called.
        """
        if type(attempt) is not int or attempt < 1 or not authorization_id.strip():
            raise ValueError("An exact attempt and nonempty approval identity are required")
        with self._transaction() as db:
            grant = db.execute(
                "SELECT * FROM retry_authorizations WHERE request_id=? AND number=?",
                (key, attempt),
            ).fetchone()
            if grant is not None:
                if grant["authorization_id"] != authorization_id:
                    raise ValueError("Retry approval identity changed")
                return
            row = db.execute(
                "SELECT * FROM attempts WHERE request_id=? ORDER BY number DESC LIMIT 1",
                (key,),
            ).fetchone()
            if row is None or row["number"] != attempt or row["state"] not in {"sent", "unknown"}:
                raise ValueError("Approval must name the latest unresolved attempt")
            if row["state"] == "sent":
                if row["host"] != socket.gethostname():
                    raise ValueError("Cannot establish that a remote sender exited")
                try:
                    os.kill(int(row["pid"]), 0)
                except ProcessLookupError:
                    db.execute(
                        "UPDATE attempts SET state='unknown',error='sender exited' "
                        "WHERE request_id=? AND number=?",
                        (key, attempt),
                    )
                else:
                    raise ValueError("Sender is still alive")
            db.execute(
                "INSERT INTO retry_authorizations VALUES (?,?,?,NULL)",
                (key, attempt, authorization_id),
            )

    def received(
        self,
        key: str,
        number: int,
        raw: bytes,
        status: int,
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Commit exact response bytes before decoding/parsing; errors remain retry records."""
        _validate_elapsed(elapsed_seconds)
        state = "completed" if 200 <= status < 300 else "rejected"
        with self._transaction() as db:
            changed = db.execute(
                "UPDATE attempts SET state=?,raw=?,sha256=?,status=?,elapsed_seconds=? "
                "WHERE request_id=? AND number=? AND state='sent'",
                (state, raw, hashlib.sha256(raw).hexdigest(), status, elapsed_seconds, key, number),
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

    def unknown(
        self, key: str, number: int, error: str, *, elapsed_seconds: float | None = None
    ) -> None:
        """Mark ambiguous transport failure; retries retain this possibly charged attempt."""
        _validate_elapsed(elapsed_seconds)
        with self._transaction() as db:
            db.execute(
                "UPDATE attempts SET state='unknown',error=?,elapsed_seconds=? "
                "WHERE request_id=? AND number=? AND state='sent'",
                (error, elapsed_seconds, key, number),
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
                    "measured_attempts": 0,
                    "unmeasured_attempts": 0,
                    "elapsed_seconds_total": None,
                },
            )
            totals["attempts"] += 1
            if row["elapsed_seconds"] is None:
                totals["unmeasured_attempts"] += 1
            else:
                totals["measured_attempts"] += 1
                totals["elapsed_seconds_total"] = (totals["elapsed_seconds_total"] or 0.0) + row[
                    "elapsed_seconds"
                ]
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
