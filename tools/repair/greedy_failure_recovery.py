"""Retain completed rows and recover the diagnosed canonical keep lookup failure."""

import math
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.primary_failure_recovery import prior_costs
from tools.repair.shared_release import authenticate, bound, validate_completion


def recovery_state(manifest):
    recovery = bound(manifest["greedy_boundary_recovery"])
    prior = bound(recovery["manifest"])
    if prior.get("greedy_boundary_recovery"):
        raise ValueError("A further same-cause greedy recovery is not admitted")
    mutable = {"dispatch_nonce", "source_commit", "capacity_admission", "greedy_boundary_recovery"}
    if recovery.get("schema") != "exact-repair/primary-greedy-boundary-recovery/v1" or {
        k: v for k, v in prior.items() if k not in mutable
    } != {k: v for k, v in manifest.items() if k not in mutable}:
        raise ValueError("Greedy recovery changed scientific inputs or budgets")
    # Authenticate the already charged metadata recovery rather than resetting it.
    prior_costs(prior)
    completion, outputs, _, job = validate_completion(recovery["receipt"])
    retirement = bound(recovery["retirement"])
    if (
        completion["status"] != "failed"
        or recovery["manifest"]["path"] not in job["commands"][0]
        or retirement.get("retired_step") != completion["step_id"]
        or retirement.get("all_owned_processes_retired") is not True
        or completion["step_id"] in retirement["live_steps"]
        or retirement["observed_epoch"] < completion["finished_epoch"]
    ):
        raise ValueError("Greedy recovery lacks a retired failed owner")
    work = Path(completion["work"])
    report = bound(dict(path=str(work / "report.json"), sha256=outputs["report.json"]))
    identity = canonical_hash(prior)
    if report["manifest_identity"] != identity or [r["id"] for r in report["rows"]] != [
        r["id"] for r in prior["rows"]
    ]:
        raise ValueError("Greedy recovery changed the report denominator")
    costs, retained = {}, {}
    unattempted = False
    for declaration, row in zip(prior["rows"], report["rows"]):
        if row["status"] == "not_attempted":
            unattempted = True
            continue
        if unattempted or costs:
            raise ValueError("Greedy recovery requires a terminal failed prefix")
        relative = "rows/" + row["id"] + "/completion.json"
        reference = dict(path=str(work / relative), sha256=outputs[relative])
        saved = bound(reference)
        if (
            saved != {k: v for k, v in row.items() if k != "id"}
            or saved["identity"] != canonical_hash((identity, declaration))
            or saved.get("cleanup_complete") is not True
            or not math.isfinite(saved["charged_seconds"])
            or saved["charged_seconds"] < 0
        ):
            raise ValueError("Greedy recovery row receipt or cleanup differs")
        for artifact in saved.get("artifacts", []):
            path = Path(artifact["path"])
            if (
                not path.is_relative_to(work)
                or outputs.get(str(path.relative_to(work))) != artifact["sha256"]
            ):
                raise ValueError("Retained row artifact is not a terminal output")
            authenticate(artifact)
        if saved["status"] == "error":
            if (
                declaration["adapter"]["operation"] != "observable_score_greedy"
                or saved.get("detail") != "StopIteration: "
                or saved.get("result") is not None
                or saved["charged_seconds"] >= declaration["case_seconds"] - 5
                or row["id"] != recovery["failed_row_id"]
            ):
                raise ValueError("Not the diagnosed bounded greedy keep lookup failure")
            costs[row["id"]] = saved["charged_seconds"]
        elif saved["status"] in {"complete", "timeout"}:
            retained[row["id"]] = dict(
                saved,
                identity=canonical_hash((canonical_hash(manifest), declaration)),
                retained_row=reference,
                prior_attempt=recovery["receipt"],
                cost_rule="Completed scientific row retained once, never replayed",
            )
        else:
            raise ValueError("Unrecognized retained scientific row status")
    if len(costs) != 1:
        raise ValueError("Greedy recovery requires exactly one diagnosed failure")
    return costs, retained
