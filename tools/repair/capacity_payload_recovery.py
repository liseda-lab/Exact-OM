"""Retain an authenticated DEV prefix after unpublished payload hashing failed.

No expired row is replayed and no result is reconstructed from partial files.
The old cleanup lineage, row costs and original scientific manifest stay bound.
"""

from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.scaling_payload_recovery import payload_manifest
from tools.repair.shared_release import authenticate, bound, validate_completion


def retained_rows(manifest, recovery):
    from tools.repair.primary_evaluation import validate_saved
    from tools.repair.representative_capacity import recovery_rows

    prior = bound(recovery["manifest"])
    changed = {"dispatch_nonce", "concurrent_barrier", "technical_recovery"}
    if (
        {k: v for k, v in prior.items() if k not in changed}
        != {k: v for k, v in manifest.items() if k not in changed}
        or recovery.get("budgets_reset") is not False
        or recovery.get("history_reset") is not False
        or recovery.get("failed_case_replayed") is not False
    ):
        raise ValueError("Payload recovery changed scientific rows, limits or history")
    completion, outputs, _, job = validate_completion(recovery["receipt"])
    work = Path(completion["work"])
    failed = recovery["failed_index"]
    if type(failed) is not int or not 0 <= failed < len(prior["rows"]):
        raise ValueError("Invalid interrupted row index")
    row = prior["rows"][failed]
    directory = work / "rows" / row["id"]
    temporary = recovery["temporary"]
    if (
        completion["status"] != "failed"
        or recovery["manifest"]["path"] not in job["commands"][0]
        or not completion.get("error", {})
        .get("message", "")
        .startswith("command 0 exited 1: PermissionError:")
        or temporary["path"] not in completion["error"]["message"]
        or Path(temporary["path"]).parent != directory / "payload"
        or temporary.get("reason") != "unpublished_atomic_write_temporary"
        or temporary.get("mode") != 0
        or (directory / "completion.json").exists()
    ):
        raise ValueError("Not the diagnosed unpublished-payload failure")
    retirement = bound(recovery["retirement"])
    if (
        retirement.get("retired_step") != completion["step_id"]
        or completion["step_id"] in retirement["live_steps"]
        or retirement["observed_epoch"] < completion["finished_epoch"]
        or retirement.get("all_owned_processes_retired") is not True
    ):
        raise ValueError("Failed worker ownership has not retired")
    published, unpublished = payload_manifest(work)
    if (
        {str(Path(r["path"]).relative_to(work)): r["sha256"] for r in published} != outputs
        or unpublished != bound(recovery["unpublished_inventory"])["files"]
        or temporary not in unpublished
        or read(work / "identity.json") != dict(identity=canonical_hash(prior))
    ):
        raise ValueError("Original output inventory or identity changed")
    # Authenticate the previous cleanup recovery too; do not reset its lineage.
    ancestors = recovery_rows(prior)
    debris = {item["path"]: item for item in unpublished}
    retained = {}
    for index, frozen in enumerate(prior["rows"]):
        path = work / "rows" / frozen["id"] / "completion.json"
        new_identity = canonical_hash((canonical_hash(manifest), frozen))
        if index > failed:
            if path.parent.exists():
                raise ValueError("Continuation row already spent work")
            continue
        if index == failed:
            guard = path.parent / "started.json"
            started = read(guard)
            if (
                outputs.get(str(guard.relative_to(work))) != binding(guard)["sha256"]
                or started["identity"] != canonical_hash((canonical_hash(prior), frozen))
                or started["cost_identity"] != frozen["id"]
                or started["reserved_seconds"] != frozen["case_seconds"]
                or not started["started_epoch"]
                < started["deadline_epoch"]
                <= completion["finished_epoch"]
            ):
                raise ValueError("Interrupted row guard or expired reservation differs")
            retained[frozen["id"]] = dict(
                identity=new_identity,
                status="prior_implementation_failure",
                result=None,
                detail="No committed row outcome; unpublished payload hashing failed; expired row never replayed",
                cleanup_complete=True,
                prior_cleanup_complete=None,
                charged_seconds=started["reserved_seconds"],
                prior_charged_seconds=started["reserved_seconds"],
                reserved_seconds=started["reserved_seconds"],
                artifacts=[
                    binding(guard),
                    recovery["retirement"],
                    recovery["unpublished_inventory"],
                ],
                unpublished_artifacts=[temporary],
                prior_attempt=recovery["receipt"],
                shared_cost_reference=frozen["adapter"]["pool_dependency"],
                cost_rule="Prior interrupted row charged once at full original reservation; no result inferred from partial files",
            )
            continue
        if outputs.get(str(path.relative_to(work))) != binding(path)["sha256"]:
            raise ValueError("Retained row is not a bound terminal output")
        saved = read(path)
        if frozen["id"] in ancestors and saved != ancestors[frozen["id"]]:
            raise ValueError("Earlier recovery lineage changed")
        # Legacy receipts included readable atomic-write debris. Keep the exact
        # old receipt as evidence but never promote that debris into new outputs.
        artifacts = []
        for artifact in saved.get("artifacts", []):
            if artifact["path"] not in debris:
                authenticate(artifact)
                artifacts.append(artifact)
        checked = dict(saved, artifacts=artifacts)
        validate_saved(checked, canonical_hash((canonical_hash(prior), frozen)))
        if saved["status"] in {
            "error",
            "interrupted_unknown",
            "cleanup_incomplete",
            "not_attempted",
        }:
            raise ValueError("Prefix contains another unresolved implementation failure")
        retained[frozen["id"]] = dict(
            checked,
            identity=new_identity,
            artifacts=artifacts + [binding(path)],
            unpublished_artifacts=[
                item for item in unpublished if Path(item["path"]).is_relative_to(path.parent)
            ],
            retained_from=recovery["receipt"],
            retained_row=binding(path),
            prior_charged_seconds=saved["charged_seconds"],
        )
    if not set(ancestors) <= set(retained):
        raise ValueError("Earlier failed rows disappeared from the denominator")
    return retained
