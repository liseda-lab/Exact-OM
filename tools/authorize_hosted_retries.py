"""Apply a checksum-bound, request-specific approval to a copied hosted ledger."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from exact.llm.ledger import RequestLedger
from tools.prepared_batch import read, verified, write


def apply_authorizations(recipe, runtime):
    runtime = Path(runtime)
    expected = Path(recipe["root"]) / "runtime" / recipe["campaign_id"]
    if runtime.resolve() != expected.resolve() or not (runtime / "account-import.json").is_file():
        raise ValueError("Retry approval requires this recovery's copied account")
    binding = recipe["hosted_retry_authorizations"]
    approval = read(verified(binding))
    if (
        approval.get("schema_version") != 1
        or approval.get("approved_by") != "user"
        or approval.get("scientific_step") != recipe["scientific_step"]
        or approval.get("scope") != "exactly_one_additional_wire_attempt"
        or not approval.get("user_message")
        or len(approval.get("requests", [])) != 1
    ):
        raise ValueError("Expected one explicit request-specific user approval")
    request = approval["requests"][0]
    path = runtime / "openrouter/requests.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT a.*, r.tokens, q.identity FROM attempts a "
            "JOIN reservations r USING(request_id,number) "
            "JOIN requests q USING(request_id) WHERE a.request_id=? AND a.number=?",
            (request["request_id"], request["attempt"]),
        ).fetchone()
    if (
        row is None
        or row["tokens"] != request["reserved_tokens"]
        or row["started"] != request["started"]
        or row["host"] != request["host"]
        or row["pid"] != request["pid"]
        or row["state"] not in {"sent", "unknown"}
        or any(row[name] is not None for name in ("raw", "sha256", "status", "usage"))
        or hashlib.sha256(row["identity"].encode()).hexdigest() != request["request_id"]
    ):
        raise ValueError("Unresolved request or retained reservation differs from approval")
    RequestLedger(path.parent).authorize_unknown_once(
        request["request_id"], attempt=request["attempt"], authorization_id=binding["sha256"]
    )
    write(
        runtime / "retry-authorization.json",
        {"approval": binding, "request": request, "original_charge_retained": True},
        immutable=True,
    )
