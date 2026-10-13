"""Opt-in notifications for an isolated validation while campaign execution stays paused."""

import hashlib
import json
from pathlib import Path
import re

from .notifications import notify_intervention

_ACTIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
_ABSENT = {"MISSING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
_MAX_RECEIPT_BYTES = 8 * 1024**2


def _json(path, expected=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(_MAX_RECEIPT_BYTES + 1)
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ValueError("Validation receipt exceeds the monitoring size bound")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Validation receipt checksum changed")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Validation receipt must be a JSON object")
    json.dumps(value, allow_nan=False)  # Reject NaN/Infinity, including nested output metrics.
    return value


def _bound_json(binding):
    if (
        not isinstance(binding, dict)
        or not isinstance(binding.get("path"), str)
        or not Path(binding["path"]).is_absolute()
        or re.fullmatch(r"[a-f0-9]{64}", str(binding.get("sha256"))) is None
    ):
        raise ValueError("Validation receipt requires an absolute path and SHA256 binding")
    return _json(binding["path"], binding["sha256"])


def notify_paused_reviews(directory, policy, registry, observation, state, *, now):
    """Queue one action-required message per explicitly admitted validation attempt.

    This is called only by the paused branch. It observes receipts and updates its
    own check counters; it cannot clear a pause, submit, repair or resume a job.
    The ordinary notification worker owns delivery, even while execution is paused.
    """
    allowed = registry.get("paused_monitor_allowlist", [])
    if not isinstance(allowed, list):
        return []
    findings = {row["run_id"]: row for row in observation.get("findings", [])}
    alerts = []
    for run in registry["runs"]:
        identity = {"run_id": run["id"], "dispatch_nonce": run.get("dispatch_nonce")}
        if (
            identity not in allowed
            or not run.get("enabled", True)
            or run.get("require_result_review") is not True
            or run.get("selection_eligible") is not False
        ):
            continue
        finding = findings.get(run["id"], {})
        step, nonce = run["step_id"], run.get("dispatch_nonce", "")
        if (
            not isinstance(step, str)
            or not isinstance(nonce, str)
            or not isinstance(policy.get("allocation"), str)
            or finding.get("step_id") != step
            or re.fullmatch(re.escape(policy["allocation"]) + r"\.\d+", step) is None
            or re.fullmatch(r"[a-f0-9]{16,64}", nonce) is None
        ):
            continue
        attempt = hashlib.sha256(json.dumps([identity, step], sort_keys=True).encode()).hexdigest()[:24]
        observed = state.setdefault("paused_reviews", {}).setdefault(attempt, {})
        scheduler = finding.get("scheduler_state")
        if scheduler in _ACTIVE:
            if scheduler == "RUNNING":
                observed["seen_running"] = True
            observed.pop("absent_since", None)
            observed.pop("absent_checks", None)
        elif scheduler in _ABSENT and observed.get("seen_running"):
            observed.setdefault("absent_since", now)
            observed["absent_checks"] = observed.get("absent_checks", 0) + 1
        else:
            observed.pop("absent_since", None)
            observed.pop("absent_checks", None)

        summary = None
        error = "The validation step disappeared without an owned terminal receipt."
        try:
            recipe = run["review_recipe"]
            _bound_json(recipe)
            receipt = _json(run["completion_path"])
            if (
                receipt.get("step_id") != step
                or receipt.get("dispatch_nonce") != nonce
                or receipt.get("recipe_sha256") != recipe["sha256"]
            ):
                raise ValueError("Validation completion belongs to another step, attempt or recipe")
            code = receipt.get("exit_code")
            if type(code) is not int:
                raise ValueError("Validation completion lacks an integer exit code")
            if receipt.get("status") == "failed" and code != 0:
                summary = "The isolated validation failed. Review its retained error and results before approving any further work."
            elif receipt.get("status") == "complete" and code == 0:
                report = receipt["review_report"]
                if report["path"] != run["review_report_path"]:
                    raise ValueError("Validation completion names an unapproved result path")
                _bound_json(report)
                if scheduler in _ABSENT:
                    summary = "The isolated validation finished. Review its results and throughput before approving any scientific restart. Completion does not establish scientific admission."
            else:
                raise ValueError("Validation completion is not a terminal success or failure")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error = "Validation completion could not be verified: " + type(exc).__name__ + ": " + str(exc)
        if summary is None and (
            observed.get("absent_checks", 0) >= 2
            and now - observed.get("absent_since", now) >= 60
        ):
            summary = error + " Inspect the retained receipts; no automatic retry is authorized."
        if summary is None:
            continue
        summary += " G4 and all other experiments remain paused; this alert does not permit dispatch or recovery."
        alerts.append(notify_intervention(
            directory,
            {
                "id": "paused-validation-review-" + attempt,
                "kind": "validation_review",
                "run_ids": [run["id"]],
                "reason": "The single authorized validation needs operator review.",
            },
            "approval_needed",
            summary,
            config=policy.get("notifications", {}),
            handoff=run.get("handoff", run.get("review_report_path", "")),
            defer=True,
        ))
    return alerts
