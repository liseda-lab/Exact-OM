"""One campaign-wide admission authority, independent of copied response ledgers.

Reservations commit before local send claims. A crash between those transactions
retains conservative exposure; only explicit reconciliation may release an orphan.
Workers never initialize or reset the shared store.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from exact.llm.prompt_budget import PromptBudgetError
from exact.utils.hosted_spending import load_spending_policy

CAMPAIGN_ENV = "EXACT_HOSTED_CAMPAIGN_ID"
EXPERIMENT_ENV = "EXACT_HOSTED_EXPERIMENT_ID"
MAX_SQL_INTEGER = 2**63 - 1


class HostedSpendingError(PromptBudgetError, ValueError):
    """Paid admission cannot establish trustworthy accounting; fail closed."""


class HostedSpendPause(HostedSpendingError):
    """A durable paid-work pause requiring an explicitly increased allowance."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__(
            f"Hosted spending paused: {details['scope']} "
            f"{details.get('experiment_id') or details['campaign_id']}; "
            f"accounted={details['accounted_tokens']}, next_reserved={details['requested_tokens']}, "
            f"limit={details['limit']}; explicit allowance amendment required"
        )


def selected_policy(binding: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Load either policy version while retaining typed caller propagation."""
    try:
        return load_spending_policy(binding)
    except (OSError, ValueError, TypeError) as exc:
        raise HostedSpendingError(f"Cannot verify the bound hosted spending policy: {exc}") from exc


def _hard_policy(binding: Mapping[str, Any] | None) -> dict[str, Any]:
    selected = selected_policy(binding)
    if selected is None or selected["policy"]["schema_version"] != 2:
        raise HostedSpendingError("A bound v2 hosted spending policy is required")
    return selected


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SQL_INTEGER:
        raise HostedSpendingError(f"Invalid nonnegative accounting integer: {label}")
    return value


def _cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _bootstrap(selected: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    policy = selected["policy"]
    record = policy["bootstrap"]
    try:
        raw = Path(record["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise HostedSpendingError("Hosted spending bootstrap binding changed")
        value = json.loads(raw)
        if (
            type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("campaign_id") != policy["campaign_id"]
        ):
            raise HostedSpendingError("Hosted spending bootstrap campaign/schema mismatch")
        total = _integer(value.get("accounted_tokens"), "bootstrap total")
        experiments = value.get("experiment_tokens")
        if not isinstance(experiments, dict) or any(
            not isinstance(name, str) or not name.strip() for name in experiments
        ):
            raise HostedSpendingError("Bootstrap requires stable experiment attribution")
        if sum(_integer(amount, name) for name, amount in experiments.items()) > total:
            raise HostedSpendingError("Bootstrap experiment attribution exceeds campaign total")
        if _cost(value.get("reported_usd", 0.0)) is None:
            raise HostedSpendingError("Invalid bootstrap reported USD")
        _integer(value.get("unpriced_attempts", 0), "bootstrap unpriced attempts")
        if not isinstance(value.get("sources"), list) or not value["sources"]:
            raise HostedSpendingError("Bootstrap requires immutable accounting source bindings")
        for source in value["sources"]:
            if (
                not isinstance(source, dict)
                or not isinstance(source.get("path"), str)
                or not Path(source["path"]).is_absolute()
                or not isinstance(source.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
            ):
                raise HostedSpendingError("Invalid historical accounting source binding")
        return raw, value
    except HostedSpendingError:
        raise
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise HostedSpendingError("Cannot read the bound historical spending bootstrap") from exc


def _verify_store(db: sqlite3.Connection, selected: Mapping[str, Any]) -> dict[str, Any]:
    row = db.execute("SELECT record FROM metadata WHERE singleton=1").fetchone()
    if row is None:
        raise HostedSpendingError("Shared spending store is not initialized")
    metadata: dict[str, Any] = json.loads(row[0])
    policy = selected["policy"]
    if metadata.get("schema_version") != 1:
        raise HostedSpendingError("Unsupported shared spending store schema")
    retained = metadata["bootstrap_json"].encode("utf-8")
    if (
        hashlib.sha256(retained).hexdigest() != policy["bootstrap"]["sha256"]
        or json.loads(retained) != metadata["bootstrap"]
    ):
        raise HostedSpendingError("Retained spending bootstrap bytes changed")
    if any(
        metadata.get(key) != expected
        for key, expected in (
            ("campaign_id", policy["campaign_id"]),
            ("admission_store", policy["admission_store"]),
            ("bootstrap_sha256", policy["bootstrap"]["sha256"]),
        )
    ):
        raise HostedSpendingError("Shared spending store identity differs from bound policy")
    return metadata


@contextmanager
def _transaction(
    selected: Mapping[str, Any]
) -> Iterator[tuple[sqlite3.Connection, dict[str, Any]]]:
    path = Path(selected["policy"]["admission_store"])
    try:
        # r+ and mode=rw deliberately cannot initialize a missing authority.
        with path.with_suffix(path.suffix + ".lock").open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                metadata = _verify_store(db, selected)
                yield db, metadata
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
                fcntl.flock(lock, fcntl.LOCK_UN)
    except HostedSpendingError:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        raise HostedSpendingError(
            "Shared hosted admission accounting is unavailable or invalid"
        ) from exc


def initialize_admission_store(binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Explicitly import historical charges exactly once; never called by workers."""
    selected = _hard_policy(binding)
    raw, bootstrap = _bootstrap(selected)
    # Verify the provenance at bootstrap time; immutable snapshots remain retained
    # inside metadata even when the active cumulative account moves to a new run.
    for source in bootstrap["sources"]:
        try:
            data = Path(source["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise HostedSpendingError("Historical spending source binding changed")
        except (OSError, TypeError, KeyError) as exc:
            raise HostedSpendingError("Cannot verify historical spending provenance") from exc
    path = Path(selected["policy"]["admission_store"])
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_suffix(path.suffix + ".initialized.json")
    marker_text = json.dumps(
        {
            "admission_store": str(path),
            "bootstrap_sha256": selected["policy"]["bootstrap"]["sha256"],
        },
        sort_keys=True,
    )
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists() and marker.read_text() != marker_text:
            raise HostedSpendingError("Shared spending namespace initialization marker changed")
        if not path.exists() and marker.exists():
            raise HostedSpendingError(
                "Previously initialized spending store is missing; reconciliation required"
            )
        if not path.exists():
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            db = sqlite3.connect(path)
            try:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE metadata(singleton INTEGER PRIMARY KEY,record TEXT NOT NULL)"
                )
                db.execute(
                    """CREATE TABLE grants(
                    id TEXT PRIMARY KEY,experiment_id TEXT NOT NULL,request_id TEXT NOT NULL,
                    local_ledger TEXT NOT NULL,attempt INTEGER NOT NULL,policy_sha256 TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL,accounted_tokens INTEGER NOT NULL,
                    usage TEXT,cost REAL,created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
                )
                db.execute(
                    """CREATE TABLE pauses(
                    id TEXT PRIMARY KEY,scope TEXT NOT NULL,experiment_id TEXT,
                    record TEXT NOT NULL,created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
                )
                metadata = {
                    "schema_version": 1,
                    "campaign_id": bootstrap["campaign_id"],
                    "admission_store": str(path),
                    "bootstrap_sha256": selected["policy"]["bootstrap"]["sha256"],
                    "bootstrap": bootstrap,
                    "bootstrap_json": raw.decode("utf-8"),
                }
                db.execute(
                    "INSERT INTO metadata VALUES(1,?)", (json.dumps(metadata, sort_keys=True),)
                )
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
        with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as verified:
            _verify_store(verified, selected)
        if not marker.exists():
            with marker.open("x") as stream:
                stream.write(marker_text)
                stream.flush()
                os.fsync(stream.fileno())
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return admission_snapshot(binding)


def _experiment_cap(policy: Mapping[str, Any], experiment: str) -> int:
    return int(
        policy.get("experiment_tokens_caps", {}).get(experiment, policy["experiment_tokens_cap"])
    )


def _snapshot(
    db: sqlite3.Connection, metadata: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    policy, baseline = selected["policy"], metadata["bootstrap"]
    experiments = {
        key: {
            "accounted_tokens": amount,
            "historical_tokens": amount,
            "reserved_tokens": 0,
            "reported_cost_usd": 0.0,
            "historical_reported_cost_usd": None,
            "unpriced_attempts": 0,
        }
        for key, amount in baseline["experiment_tokens"].items()
    }
    total = baseline["accounted_tokens"]
    reserved = 0
    cost = float(baseline.get("reported_usd", 0.0))
    unpriced = baseline.get("unpriced_attempts", 0)
    for row in db.execute(
        "SELECT experiment_id,SUM(accounted_tokens) AS accounted_tokens, "
        "SUM(CASE WHEN usage IS NULL THEN accounted_tokens ELSE 0 END) AS reserved_tokens, "
        "COALESCE(SUM(cost),0.0) AS cost,SUM(cost IS NULL) AS unpriced FROM grants "
        "GROUP BY experiment_id"
    ):
        item = experiments.setdefault(
            row["experiment_id"],
            {
                "accounted_tokens": 0,
                "historical_tokens": 0,
                "reserved_tokens": 0,
                "reported_cost_usd": 0.0,
                "unpriced_attempts": 0,
            },
        )
        amount = row["accounted_tokens"]
        outstanding = row["reserved_tokens"]
        item["accounted_tokens"] += amount
        item["reserved_tokens"] += outstanding
        item["reported_cost_usd"] += row["cost"] or 0.0
        item["unpriced_attempts"] += row["unpriced"]
        total += amount
        reserved += outstanding
        cost += row["cost"] or 0.0
        unpriced += row["unpriced"]
    pauses = []
    for row in db.execute("SELECT record FROM pauses ORDER BY created,id"):
        record = json.loads(row["record"])
        cap = (
            policy["campaign_tokens_cap"]
            if record["scope"] == "campaign"
            else _experiment_cap(policy, record["experiment_id"])
        )
        # A policy text/hash change alone is not an allowance increase.
        pauses.append({**record, "active": cap <= record["limit"]})
    paused_experiments = sorted(
        {p["experiment_id"] for p in pauses if p["active"] and p["scope"] == "experiment"}
    )
    for name, item in experiments.items():
        item["tokens_cap"] = _experiment_cap(policy, name)
        item["status"] = (
            "paused"
            if name in paused_experiments
            else (
                "over_cap"
                if item["accounted_tokens"] >= _experiment_cap(policy, name)
                else (
                    "warning"
                    if item["accounted_tokens"] >= policy["experiment_warning_tokens"]
                    else "ok"
                )
            )
        )
    return {
        "campaign_id": policy["campaign_id"],
        "policy_sha256": selected["sha256"],
        "admission_store": policy["admission_store"],
        "accounted_tokens": total,
        "historical_tokens": baseline["accounted_tokens"],
        "reserved_tokens": reserved,
        "reported_cost_usd": cost,
        "unpriced_attempts": unpriced,
        "campaign_tokens_cap": policy["campaign_tokens_cap"],
        "experiment_tokens_cap": policy["experiment_tokens_cap"],
        "experiment_tokens_caps": policy.get("experiment_tokens_caps", {}),
        "experiment_warning_tokens": policy["experiment_warning_tokens"],
        "notification_tokens": policy["notification_tokens"],
        "experiments": experiments,
        "paused": any(p["active"] and p["scope"] == "campaign" for p in pauses),
        "paused_experiments": paused_experiments,
        "pauses": pauses,
    }


def admission_snapshot(binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read shared totals and actual pause events; historical excess is not an event."""
    selected = _hard_policy(binding)
    with _transaction(selected) as (db, metadata):
        return _snapshot(db, metadata, selected)


def reserve_attempt(
    selected: Mapping[str, Any], *, tokens: int, request_id: str, local_ledger: str, attempt: int
) -> str:
    """Atomically check both scopes and commit exposure before any local sent row."""
    policy = selected["policy"]
    campaign = os.getenv(CAMPAIGN_ENV)
    experiment = os.getenv(EXPERIMENT_ENV)
    if (
        campaign != policy["campaign_id"]
        or not experiment
        or not re.fullmatch(r"E[0-9]{2}|G0", experiment)
    ):
        raise HostedSpendingError(
            "Paid requests require bound campaign and canonical experiment scope"
        )
    if _integer(tokens, "reservation") == 0:
        raise HostedSpendingError("Paid request reservation must be positive")
    denial = None
    identifier = uuid.uuid4().hex
    with _transaction(selected) as (db, metadata):
        current = _snapshot(db, metadata, selected)
        scopes = (
            ("campaign", None, current["accounted_tokens"], policy["campaign_tokens_cap"]),
            (
                "experiment",
                experiment,
                current["experiments"].get(experiment, {}).get("accounted_tokens", 0),
                _experiment_cap(policy, experiment),
            ),
        )
        for scope, name, used, limit in scopes:
            retained = next(
                (
                    p
                    for p in current["pauses"]
                    if p["active"] and p["scope"] == scope and p["experiment_id"] == name
                ),
                None,
            )
            if retained is not None:
                denial = retained
                break
            if used + tokens > limit:
                denial = {
                    "id": uuid.uuid4().hex,
                    "scope": scope,
                    "experiment_id": name,
                    "campaign_id": campaign,
                    "accounted_tokens": used,
                    "requested_tokens": tokens,
                    "limit": limit,
                    "policy_sha256": selected["sha256"],
                    "ledger_path": local_ledger,
                    "request_id": request_id,
                    "attempt": attempt,
                }
                db.execute(
                    "INSERT INTO pauses(id,scope,experiment_id,record) VALUES(?,?,?,?)",
                    (denial["id"], scope, name, json.dumps(denial, sort_keys=True)),
                )
                break
        if denial is None:
            db.execute(
                """INSERT INTO grants(id,experiment_id,request_id,local_ledger,attempt,
                          policy_sha256,reserved_tokens,accounted_tokens) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    experiment,
                    request_id,
                    local_ledger,
                    attempt,
                    selected["sha256"],
                    tokens,
                    tokens,
                ),
            )
    # Raise only after committing the denied-admission event, never a sent claim.
    if denial is not None:
        raise HostedSpendPause(denial)
    return identifier


def settle_attempt(selected: Mapping[str, Any], grant_id: str, usage: Mapping[str, Any]) -> None:
    """Replace exposure only with complete valid usage; preserve missing/unknown cost."""
    known = {
        key: usage[key]
        for key in ("prompt_tokens", "completion_tokens")
        if type(usage.get(key)) is int and 0 <= usage[key] <= MAX_SQL_INTEGER
    }
    partial = sum(known.values())
    if partial > MAX_SQL_INTEGER:
        raise HostedSpendingError("Provider usage exceeds accounting integer range")
    complete = len(known) == 2
    cost = _cost(usage.get("cost"))
    encoded = json.dumps(known, sort_keys=True) if complete else None
    with _transaction(selected) as (db, _metadata):
        row = db.execute(
            "SELECT usage,cost,accounted_tokens FROM grants WHERE id=?", (grant_id,)
        ).fetchone()
        if row is None:
            raise HostedSpendingError("Hosted attempt lacks its retained central grant")
        if row["usage"] is not None and encoded != row["usage"]:
            raise HostedSpendingError("Settled hosted usage cannot be rewritten")
        if row["cost"] is not None and cost is not None and cost != row["cost"]:
            raise HostedSpendingError("Reported hosted cost cannot be rewritten")
        retained_cost = row["cost"] if row["cost"] is not None else cost
        amount = partial if complete else max(row["accounted_tokens"], partial)
        db.execute(
            "UPDATE grants SET accounted_tokens=?,usage=?,cost=? WHERE id=?",
            (amount, encoded, retained_cost, grant_id),
        )
