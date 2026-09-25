"""Read-only, bounded experiment health checks; never infer failure from log silence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_ACTIVE = {
    "RUNNING",
    "PENDING",
    "CONFIGURING",
    "COMPLETING",
    "RESIZING",
    "SIGNALING",
    "SUSPENDED",
    "STAGE_OUT",
}
_TERMINAL = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}
_FAILED = {"blocked", "failed", "interrupted", "stopping", "deferred_budget"}
_VOLATILE = {"recorded_at", "updated_at", "timestamp", "started_at", "ended_at", "pid"}
_MAX_METADATA_BYTES = 1024 * 1024


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\d{4}-\d{2}-\d{2}[T ][0-9:.]+(?:Z|[+-]\d\d:\d\d)?", "<time>", value)
    return value


def _incident(run_id: str | None, kind: str, reason: str, detail: Any = None) -> dict:
    identity = json.dumps([run_id, kind, _stable(detail)], sort_keys=True)
    return {
        "id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "kind": kind,
        "focus_run_id": run_id,
        "run_ids": [run_id] if run_id else [],
        "reason": reason,
    }


def _registry(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ids = [run.get("id") for run in runs]
    if any(not isinstance(name, str) or not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("Run IDs must be nonempty and unique")
    for run in runs:
        dependencies = run.get("depends_on", [])
        if not isinstance(dependencies, list) or any(name not in ids for name in dependencies):
            raise ValueError(f"Invalid dependencies for {run['id']}")
        if not isinstance(run.get("step_id"), str) or not run["step_id"]:
            raise ValueError(f"Missing step_id for {run['id']}")
    by_id = {run["id"]: run for run in runs}

    def visit(name: str, path: set[str]) -> None:
        if name in path:
            raise ValueError("Run dependencies contain a cycle")
        for parent in by_id[name].get("depends_on", []):
            visit(parent, path | {name})

    for name in ids:
        visit(str(name), set())
    return [run for run in runs if run.get("enabled", True)]


def _read(path: str | Path | None, *, json_object: bool) -> Any:
    if not path:
        return None
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(_MAX_METADATA_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds 1 MiB")
    if json_object:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("metadata must contain an object")
        return value
    return int(raw.strip())


def inspect_runs(
    runs: Sequence[Mapping[str, Any]], *, step_states: Mapping[str, str] | None
) -> dict:
    """Read only declared receipt paths, then assess an authoritative Slurm snapshot.

    ``None`` means the scheduler query failed; an absent key in a successful
    snapshot means the step is no longer live. No scheduler query runs here.
    """
    observations = {}
    for run in _registry(runs):
        observation: dict[str, Any] = {"errors": []}
        for field, key in (
            ("status", "status_path"),
            ("completion", "completion_path"),
            ("exit_code", "exit_path"),
        ):
            try:
                observation[field] = _read(run.get(key), json_object=field != "exit_code")
            except (OSError, ValueError, UnicodeError) as exc:
                observation["errors"].append(f"{field}: {type(exc).__name__}: {exc}")
        observations[run["id"]] = observation
    return assess_runs(runs, observations, step_states=step_states)


def assess_runs(
    runs: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    step_states: Mapping[str, str] | None,
) -> dict:
    """Pure health assessment with stable incident IDs and upstream deduplication."""
    enabled = _registry(runs)
    findings = {}
    incidents = {}
    for run in enabled:
        name = run["id"]
        observed = observations.get(name, {})
        state = observed.get("status") or {}
        completion = observed.get("completion")
        phase = str(state.get("status", ""))
        exit_code = observed.get("exit_code")
        scheduler = None if step_states is None else step_states.get(run["step_id"], "MISSING")
        scheduler = scheduler.split()[0].rstrip("+") if scheduler else scheduler
        alive = scheduler in _ACTIVE
        finding = {
            "run_id": name,
            "step_id": run["step_id"],
            "phase": phase,
            "scheduler_state": scheduler,
            "status": "healthy",
            "retryable": False,
        }
        incident = None
        if phase in _FAILED or phase.startswith("blocked_"):
            reason = f"{name} reports {phase}"
            incident = _incident(name, "run_failed", reason, state.get("error", phase))
        elif exit_code is not None and exit_code != 0:
            reason = f"{name} launcher exited with code {exit_code}"
            incident = _incident(name, "launcher_failed", reason, exit_code)
        elif completion is not None and (
            completion.get("status") in _FAILED
            or (isinstance(completion.get("exit_code"), int) and completion["exit_code"] != 0)
        ):
            reason = f"{name} completion receipt reports failure"
            incident = _incident(
                name,
                "completion_failed",
                reason,
                {key: completion.get(key) for key in ("status", "exit_code", "error")},
            )
        elif observed.get("errors"):
            finding.update(status="waiting", retryable=True, errors=observed["errors"])
            reason = "Receipt temporarily unreadable; retry without launching work"
        elif completion is not None and (
            completion.get("status") != "complete" or completion.get("exit_code", 0) != 0
        ):
            finding.update(status="waiting", retryable=True)
            reason = "Completion receipt is not a successful completion; retry read-only"
        elif scheduler is None or scheduler not in _ACTIVE | _TERMINAL | {"MISSING"}:
            finding.update(status="waiting", retryable=True)
            reason = "Scheduler state unavailable; retry without launching work"
        elif completion is not None:
            finding["status"] = "healthy" if alive else "complete"
            reason = (
                "Completed; Slurm step is finalizing" if alive else "Completion receipt present"
            )
        elif not alive:
            reason = f"{name} has no live Slurm step and no successful completion receipt"
            incident = _incident(name, "step_missing", reason)
        elif phase.startswith("waiting") or scheduler in {"PENDING", "SUSPENDED", "CONFIGURING"}:
            finding["status"] = "waiting"
            reason = "Live step is waiting for its dependency or scheduler"
        else:
            reason = "Slurm step is live; quiet logs are not a failure"
        if incident:
            finding.update(status="needs_attention", incident_id=incident["id"])
            incidents[name] = incident
        finding["reason"] = reason
        findings[name] = finding

    by_id = {run["id"]: run for run in enabled}

    def upstream_incident(name: str) -> str | None:
        for parent in by_id[name].get("depends_on", []):
            if parent not in by_id:
                continue
            root = upstream_incident(parent)
            if root or parent in incidents:
                return root or parent
        return None

    # Attribute explicit dependency failures to the upstream incident. Independent
    # failures retain their own identity even when an upstream run also failed.
    for run in enabled:
        name = run["id"]
        root = upstream_incident(name)
        observed = observations.get(name, {})
        error = json.dumps((observed.get("status") or {}).get("error", "")).lower()
        if root and (findings[name]["phase"].startswith("waiting") or "upstream" in error):
            incident = incidents[root]
            incident["run_ids"].append(name)
            incidents.pop(name, None)
            findings[name].update(
                status="waiting",
                incident_id=incident["id"],
                reason=f"Waiting for upstream incident on {root}",
            )
    if enabled and all(row["status"] == "complete" for row in findings.values()):
        incident = _incident(
            None,
            "next_batch",
            "All tracked runs completed; prepare the next eligible batch",
            sorted(run["id"] for run in enabled),
        )
        incident["run_ids"] = sorted(findings)
        incidents["next_batch"] = incident
    values = list(findings.values())
    overall = (
        "needs_attention"
        if incidents
        else (
            "waiting"
            if any(row["status"] == "waiting" for row in values)
            else "healthy" if any(row["status"] == "healthy" for row in values) else "complete"
        )
    )
    return {
        "status": overall,
        "findings": values,
        "incidents": list(incidents.values()),
        "retryable": any(row["retryable"] for row in values),
    }
