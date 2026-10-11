"""Deterministic, opt-in detection of a finished experiment queue."""

import hashlib
import json
from pathlib import Path


def readiness_incidents(registry, observation, *, now):
    """Opt-in evidence gates: completed processes need not mean a usable study.

    The registry declares exact required values, never favourable model scores.
    An explicitly queued/running recovery owns the issue until it finishes.
    Stable gate IDs keep unsuccessful repair counts across replacement jobs.
    """
    from .supervision import batch_schedule

    findings = {r["run_id"]: r for r in observation.get("findings", [])}
    pending = {r["id"]: r for r in registry.get("pending_batches", [])}
    incidents = []
    for gate in registry.get("readiness_gates", []):
        if not gate.get("enabled", True):
            continue
        recovering = gate.get("recovery_run_ids", [])
        if any(
            findings.get(r, {}).get("status") == "healthy"
            or (
                r in pending
                and pending[r].get("enabled", True)
                and batch_schedule(pending[r], now) == "open"
            )
            for r in recovering
        ):
            continue
        try:
            data = Path(gate["report"]["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != gate["report"]["sha256"]:
                raise ValueError("Readiness report integrity changed")
            report = json.loads(data)
            failures = {
                k: report.get(k)
                for k, expected in gate["required_values"].items()
                if report.get(k) != expected
            }
            if not failures:
                continue
            reason = "Campaign readiness failed: " + json.dumps(failures, sort_keys=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = "Campaign readiness evidence unavailable: " + str(error)
        incidents.append(
            dict(
                id="readiness-" + gate["id"],
                kind="campaign_readiness",
                reason=reason,
                run_ids=[],
                evidence=gate["report"],
                instructions=gate.get("instructions", "Diagnose and prepare a bounded recovery."),
            )
        )
    return incidents


def idle_completion(registry, observation, dispatch_state):
    """Require actual completed findings and no remaining or launching work."""
    findings = observation.get("findings", [])
    if (
        registry.get("remaining_work_status") not in {"terminal", "deferred"}
        or any(r.get("enabled", True) for r in registry.get("pending_batches", []))
        or observation.get("incidents")
        or not findings
        or any(row.get("status") != "complete" for row in findings)
        or any(
            row.get("scheduler_state") in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}
            for row in findings
        )
        or any(row.get("status") in {"reserved", "starting"} for row in dispatch_state.values())
    ):
        return None
    # Receipt/run identities distinguish later finished queues. Repeated ticks
    # and monitor restarts produce the same outbox id and immutable email body.
    identity = {
        "campaign": registry.get("campaign"),
        "completion": registry.get("scope_completion"),
        "runs": sorted((r["run_id"], r.get("step_id")) for r in findings),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return dict(
        id="queue-complete-" + digest,
        kind="queue_complete",
        run_ids=[r["run_id"] for r in findings],
        reason="All scheduled experiments finished; the queue is empty and the supervisor is idle.",
    )
