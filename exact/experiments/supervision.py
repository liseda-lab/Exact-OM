"""Read-only, bounded experiment health checks; never infer failure from log silence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from exact.experiments.science_health import inspect_science

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


def _incident(
    run_id: str | None, kind: str, reason: str, detail: Any = None, *, scope: str | None = None
) -> dict:
    identity = json.dumps([scope or run_id, kind, _stable(detail)], sort_keys=True)
    return {
        "id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "kind": kind,
        "scope_run_id": scope or run_id,
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
    predecessors = {}
    for run in runs:
        successor = run.get("superseded_by")
        if successor is not None:
            if not isinstance(successor, str) or successor not in by_id:
                raise ValueError(f"Invalid superseded_by for {run['id']}")
            if successor in predecessors:
                raise ValueError("Recovery runs must have a single predecessor")
            predecessors[successor] = run["id"]
    for name in ids:
        seen = set()
        while name in predecessors:
            if name in seen:
                raise ValueError("Run recoveries contain a cycle")
            seen.add(name)
            name = predecessors[name]

    def visit(name: str, path: set[str]) -> None:
        if name in path:
            raise ValueError("Run dependencies contain a cycle")
        for parent in by_id[name].get("depends_on", []):
            visit(parent, path | {name})

    for name in ids:
        visit(str(name), set())
    dependencies = _effective_dependencies(runs)

    def visit_current(name: str, path: set[str]) -> None:
        if name in path:
            raise ValueError("Recovered run dependencies contain a cycle")
        for parent in dependencies[name]:
            visit_current(parent, path | {name})

    for name in dependencies:
        visit_current(name, set())
    return [run for run in runs if run.get("enabled", True)]


def _effective_dependencies(runs: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """A waiting child follows the current recovery of its declared prerequisite."""
    by_id = {run["id"]: run for run in runs}
    enabled = {run["id"] for run in runs if run.get("enabled", True)}
    result = {}
    for name in enabled:
        parents = []
        for parent in by_id[name].get("depends_on", []):
            while by_id[parent].get("superseded_by"):
                parent = by_id[parent]["superseded_by"]
            if parent in enabled:
                parents.append(parent)
        result[name] = parents
    return result


def _recovery_identities(
    runs: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[str, dict[str, str]]]:
    """Keep retries of one failure together without merging unrelated campaigns.

    Only explicitly registered recovery links and paths are aliases. Do not erase
    arbitrary path names or numbers: they can identify genuinely different errors.
    Disabled ancestors remain relevant to the retry identity.
    """
    by_id = {run["id"]: run for run in runs}
    previous = {run["superseded_by"]: run["id"] for run in runs if run.get("superseded_by")}
    result = {}
    for name in by_id:
        root = name
        while root in previous:
            root = previous[root]
        member = root
        aliases = {}
        while member:
            run = by_id[member]
            aliases[member] = "<run>"
            aliases[run["step_id"]] = "<step>"
            for field in ("status_path", "completion_path", "exit_path"):
                if run.get(field):
                    path = Path(run[field])
                    # A filesystem root is not a reliable run identity.
                    if path.is_absolute() and path.parent != Path("/"):
                        aliases[str(path.parent)] = "<run-root>"
            member = run.get("superseded_by")
        result[name] = (root, aliases)
    return result


def _recovery_detail(value: Any, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _recovery_detail(item, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [_recovery_detail(item, aliases) for item in value]
    if isinstance(value, str) and aliases:
        # Longest first ensures a root path containing a run ID is replaced as a
        # whole; boundaries avoid changing similarly named dataset paths or IDs.
        pattern = (
            r"(?<![\w.-])("
            + "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
            + r")(?![\w.-])"
        )
        return re.sub(pattern, lambda match: aliases[match.group()], value)
    return value


def _batch_failure_detail(detail: Any) -> Any:
    """Ignore only the wrapper command position, never the exit code or diagnosis."""
    if isinstance(detail, dict) and detail.get("type") == "RuntimeError":
        message = detail.get("message")
        if isinstance(message, str):
            return {**detail, "message": re.sub(
                r"^command [0-9]+(?= exited -?[0-9]+: )", "command 0", message
            )}
    return detail


def inspection_incident(runs: Sequence[Mapping[str, Any]], finding: Mapping[str, Any]) -> dict:
    """Identify unreadable evidence by its error and registered recovery lineage."""
    _registry(runs)
    name = finding["run_id"]
    scope, aliases = _recovery_identities(runs)[name]
    return _incident(
        name,
        "inspect_evidence",
        finding["reason"],
        _recovery_detail(finding.get("errors", []), aliases),
        scope=scope,
    )


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
    enabled = {run["id"] for run in _registry(runs)}
    # Disabled ancestor receipts qualify migration of already charged retries.
    # Their read errors never become active findings or authorize a new worker.
    for run in runs:
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
        scientific = (inspect_science(run, observation.get("completion"))
                      if run["id"] in enabled else dict(errors=[], failures=[]))
        observation["errors"].extend(scientific["errors"])
        observation["scientific_failures"] = scientific["failures"]
        observations[run["id"]] = observation
    return assess_runs(runs, observations, step_states=step_states)


def pending_batches(registry: Mapping[str, Any], health: Mapping[str, Any]) -> list[dict]:
    """Identify explicitly registered batches whose prerequisites and capacity are ready.

    This opt-in registry extension leaves older supervisors' all-completed rule
    unchanged. Failed unrelated runs are not prerequisites. Recovery descendants
    retain the prerequisite identity, including disabled ancestors.
    """
    runs = registry["runs"]
    _registry(runs)
    by_id = {run["id"]: run for run in runs}
    findings = {row["run_id"]: row for row in health["findings"]}
    used: dict[str, float] = {}
    for name, row in findings.items():
        if row.get("scheduler_state") in _ACTIVE:
            for key, value in by_id[name].get("resources", {}).items():
                used[key] = used.get(key, 0) + value
    result = []
    for batch in registry.get("pending_batches", []):
        if batch.get("needs_user"):
            continue
        parents: list[str | None] = []
        for name in batch.get("depends_on", []):
            if name not in by_id:
                parents.append(None)
                continue
            while by_id[name].get("superseded_by"):
                name = by_id[name]["superseded_by"]
            parents.append(findings.get(name, {}).get("status"))
        if any(status != "complete" for status in parents):
            continue
        if any(
            used.get(key, 0) + value > registry["capacity"].get(key, 0)
            for key, value in batch.get("resources", {}).items()
        ):
            continue
        incident = _incident(
            None, "next_batch", "Prepare eligible batch " + batch["id"], batch["id"]
        )
        incident["batch_id"] = batch["id"]
        result.append(incident)
    return result


def assess_runs(
    runs: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    step_states: Mapping[str, str] | None,
) -> dict:
    """Pure health assessment with stable incident IDs and upstream deduplication."""
    enabled = _registry(runs)
    identities = _recovery_identities(runs)
    by_id = {run["id"]: run for run in runs}
    predecessors = {run["superseded_by"]: run["id"] for run in runs if run.get("superseded_by")}

    def failure(name: str, kind: str, reason: str, detail: Any = None) -> dict:
        scope, aliases = identities[name]
        original = _recovery_detail(detail, aliases)
        members = [key for key, (root, _) in identities.items() if root == scope]
        completion_error = (original.get("error") if kind == "completion_failed"
                            and isinstance(original, dict) else None)
        if isinstance(completion_error, dict) and re.match(
            r"^command [0-9]+ exited -?[0-9]+: ", str(completion_error.get("message", ""))
        ) and completion_error.get("type") == "RuntimeError" and len(members) > 1:
            kind, original = "run_failed", completion_error
        if kind != "run_failed" or len(members) < 2:
            return _incident(name, kind, reason, original, scope=scope)
        normalized = _batch_failure_detail(original)
        incident = _incident(name, kind, reason, normalized, scope=scope)
        legacy, failed_replacements, first_occurrences = set(), [], []

        def owned_completion(member):
            receipt = observations.get(member, {}).get("completion") or {}
            row = by_id[member]
            return receipt if (receipt.get("step_id") == row["step_id"]
                               and row.get("dispatch_nonce") is not None
                               and receipt.get("dispatch_nonce") == row["dispatch_nonce"]
                               and receipt.get("status") in {"failed", "complete"}) else None

        for member in members:
            observed = observations.get(member, {})
            for source in (observed.get("status"), observed.get("completion")):
                candidate = _recovery_detail((source or {}).get("error"), aliases)
                if candidate is not None and _batch_failure_detail(candidate) == normalized:
                    legacy.add(_incident(member, kind, reason, candidate, scope=scope)["id"])
            completion = observed.get("completion") or {}
            candidate = _recovery_detail(completion.get("error"), aliases)
            if candidate is not None and _batch_failure_detail(candidate) == normalized:
                legacy_detail = _recovery_detail(
                    {key: completion.get(key) for key in ("status", "exit_code", "error")}, aliases
                )
                legacy.add(_incident(member, "completion_failed", reason, legacy_detail, scope=scope)["id"])
                if (member != scope and completion.get("status") == "failed"
                        and owned_completion(member)):
                    parent, prior, qualified = predecessors[member], [], True
                    while parent is not None:
                        ancestor = owned_completion(parent)
                        qualified = qualified and ancestor is not None
                        if ancestor is not None:
                            prior.append(_batch_failure_detail(_recovery_detail(ancestor.get("error"), aliases)))
                        parent = predecessors.get(parent)
                    if normalized in prior:
                        failed_replacements.append(member)
                    elif qualified:
                        first_occurrences.append(member)
        legacy.discard(incident["id"])
        if legacy:
            incident["legacy_incident_ids"] = sorted(legacy)
        if failed_replacements:
            incident["unsuccessful_recovery_run_ids"] = sorted(failed_replacements)
        if first_occurrences:
            incident["first_occurrence_recovery_run_ids"] = sorted(first_occurrences)
        return incident

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
            incident = failure(
                name,
                "run_failed",
                reason,
                state.get("error")
                or {key: state[key] for key in ("status", "reason", "message") if key in state},
            )
        elif exit_code is not None and exit_code != 0:
            reason = f"{name} launcher exited with code {exit_code}"
            incident = failure(name, "launcher_failed", reason, exit_code)
        elif completion is not None and (
            completion.get("status") in _FAILED
            or (isinstance(completion.get("exit_code"), int) and completion["exit_code"] != 0)
        ):
            reason = f"{name} completion receipt reports failure"
            incident = failure(
                name,
                "completion_failed",
                reason,
                {key: completion.get(key) for key in ("status", "exit_code", "error")},
            )
        elif observed.get("errors"):
            finding.update(status="waiting", retryable=True, errors=observed["errors"])
            reason = "Receipt temporarily unreadable; retry without launching work"
        elif observed.get("scientific_failures"):
            failures = observed["scientific_failures"]
            # One cause owns each intervention. Counts, affected row IDs and
            # changing payload paths do not reset its retry budget.
            first = min(failures, key=lambda row: json.dumps(row["signature"], sort_keys=True))
            reason = f"{name} contains a completed nested software failure: " + first["signature"]["detail"]
            incident = failure(name, "scientific_software_error", reason, first["signature"])
            finding["scientific_failures"] = failures
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
            incident = failure(name, "step_missing", reason)
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

    dependencies = _effective_dependencies(runs)

    def upstream_incident(name: str) -> str | None:
        for parent in dependencies[name]:
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
