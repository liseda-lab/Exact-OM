"""Deterministic, opt-in detection of a finished experiment queue."""

import hashlib
import json


def idle_completion(registry, observation, dispatch_state):
    """Require actual completed findings and no remaining or launching work."""
    findings = observation.get("findings", [])
    if (
        registry.get("remaining_work_status") not in {"terminal", "deferred"}
        or registry.get("pending_batches")
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
