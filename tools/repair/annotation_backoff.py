"""Deterministic, budgeted recovery of authenticated pre-generation HTTP 429s.

Each wire attempt has its own durable directory and ordinary phase reservation.
Only the terminal observation is published at the original presentation path.
No successful, invalid, abstained or unknown-delivery response is resampled.
"""

import hashlib
import json
import math
from pathlib import Path
import time

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha


def validate_policy(manifest):
    policy = manifest.get("rate_limit_policy")
    if policy is None:
        return None
    if policy.get("resume_deadline_epoch") is not None:
        deadline = policy["resume_deadline_epoch"]
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or deadline <= 0
            or deadline > manifest.get("deadline_epoch", 0)
        ):
            raise ValueError("Transport resume deadline must fit the frozen campaign deadline")
    reference = policy["authorization"]
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Rate-limit authorization changed")
    approval = read(reference["path"])
    if (
        approval.get("schema") != "exact-repair/transient-provider-authorization/v1"
        or approval.get("status") != "approved"
        or approval.get("automatic_429_recovery") is not True
        or approval.get("lineage_id") != manifest["lineage_id"]
        or approval.get("cost_ceiling_usd") != manifest["cost_ceiling_usd"]
        or not approval.get("user_instruction")
    ):
        raise ValueError("Rate-limit recovery requires matching campaign authorization")
    base, maximum = policy["base_seconds"], policy["max_seconds"]
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (base, maximum)
        )
        or not 0 < base <= maximum <= 300
    ):
        raise ValueError("Invalid bounded rate-limit backoff")
    return policy


def retry_delay(manifest, failure, index):
    """A numeric status alone cannot authorize repetition of a paid observation."""
    from exact.llm.ledger import RequestLedger

    if failure.get("http_status") != 429:
        return None
    with RequestLedger(Path(manifest["ledger_directory"]))._transaction() as db:
        row = db.execute(
            "SELECT * FROM attempts WHERE request_id=? AND number=?",
            (failure["request_id"], failure["attempt"]),
        ).fetchone()
    if row is None or row["status"] != 429 or row["state"] != "rejected":
        raise ValueError("Rate-limit retry lacks a definite rejected wire attempt")
    raw = bytes(row["raw"] or b"")
    if (
        hashlib.sha256(raw).hexdigest() != failure["response_sha256"]
        or row["sha256"] != failure["response_sha256"]
    ):
        raise ValueError("Rate-limit response provenance changed")
    body = json.loads(raw)
    if body.get("error", {}).get("code") != 429 or body.get("choices") or body.get("usage"):
        return None
    metadata = body["error"].get("metadata", {})
    retry_after = metadata.get(
        "retry_after_seconds", metadata.get("headers", {}).get("Retry-After", 0)
    )
    try:
        retry_after = float(retry_after)
    except (TypeError, ValueError):
        retry_after = 0
    if not math.isfinite(retry_after) or retry_after < 0:
        retry_after = 0
    policy = validate_policy(manifest)
    return max(
        retry_after, min(policy["max_seconds"], policy["base_seconds"] * 2 ** min(index, 20))
    )


def annotate(packet, manifest, output, *, slot_id, swapped=False, seconds=94, comparison_id=None):
    from tools.repair.corrective_semantics import (
        ConfirmedAnnotationFailure,
        _remaining_seconds,
        annotate_packet,
    )

    policy = validate_policy(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "backoff.json"
    identity = canonical_hash((packet, manifest, slot_id, swapped, comparison_id))
    state = (
        read(checkpoint)
        if checkpoint.exists()
        else dict(
            identity=identity,
            attempt=0,
            next_attempt_epoch=0,
            failures=[],
            complete=False,
            # Optional shared service window survives short owner slices. It is
            # frozen into this identity, never renewed on resume. Wire attempts
            # still consume the original request and cost ledgers individually.
            deadline_epoch=(
                policy["resume_deadline_epoch"]
                if policy and policy.get("resume_deadline_epoch")
                else time.time() + _remaining_seconds(seconds, manifest)
            ),
        )
    )
    if state["identity"] != identity:
        raise ValueError("Rate-limit resume dependencies changed")
    if state["attempt"] != len(state["failures"]):
        raise ValueError("Rate-limit attempt history changed")
    for previous in state["failures"]:
        if sha(previous["receipt"]) != previous["receipt_sha256"]:
            raise ValueError("Rate-limit rejection receipt changed")
        if read(previous["receipt"])["confirmed_failure"] != previous["evidence"]:
            raise ValueError("Rate-limit rejection evidence changed")
    if state["complete"]:
        for name, digest in state["terminal_files"].items():
            if sha(output / name) != digest:
                raise ValueError("Rate-limit terminal output changed")
        return output / "labels.json" if state["has_labels"] else None
    write_artifact(checkpoint, state)
    call_deadline = time.time() + _remaining_seconds(seconds, manifest)
    while True:
        remaining = min(call_deadline - time.time(), state["deadline_epoch"] - time.time())
        if remaining < 92 or state["next_attempt_epoch"] + 92 >= min(
            state["deadline_epoch"], call_deadline
        ):
            write_artifact(
                output / "receipt.json",
                dict(
                    status="rate_limit_deferred",
                    identity=identity,
                    next_attempt_epoch=state["next_attempt_epoch"],
                    checkpoint_sha256=sha(checkpoint),
                    costs_reset=False,
                ),
            )
            return None
        delay = state["next_attempt_epoch"] - time.time()
        if delay > 0:
            time.sleep(min(delay, 30))
            continue
        directory = output / "transport-attempts" / f"{state['attempt']:04}"
        try:
            artifact = annotate_packet(
                packet,
                manifest,
                directory,
                slot_id=f"{slot_id}:transport:{state['attempt']}",
                swapped=swapped,
                seconds=remaining,
                comparison_id=comparison_id or slot_id,
            )
        except ConfirmedAnnotationFailure as error:
            delay = retry_delay(manifest, error.evidence, state["attempt"])
            if delay is None:
                raise
            state["failures"].append(
                dict(
                    evidence=error.evidence,
                    receipt=str(directory / "receipt.json"),
                    receipt_sha256=sha(directory / "receipt.json"),
                )
            )
            state.update(attempt=state["attempt"] + 1, next_attempt_epoch=time.time() + delay)
            write_artifact(checkpoint, state)
            print(
                json.dumps(
                    dict(
                        status="waiting_provider_capacity",
                        slot=slot_id,
                        next_attempt_epoch=state["next_attempt_epoch"],
                        transport_failures=len(state["failures"]),
                    )
                ),
                flush=True,
            )
            continue
        # Publish the final observation, with every rejected predecessor retained.
        files = {}
        for name in ("packet.json", "run.json", "schedule.json", "labels.json", "receipt.json"):
            source = directory / name
            if source.exists():
                (output / name).write_bytes(source.read_bytes())
                files[name] = sha(output / name)
        state.update(complete=True, has_labels=artifact is not None, terminal_files=files)
        write_artifact(checkpoint, state)
        return output / "labels.json" if artifact is not None else None
