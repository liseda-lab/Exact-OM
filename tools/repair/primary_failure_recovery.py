"""Authenticate a pre-result implementation failure and retain its consumed budget."""

from pathlib import Path
import math

from exact.repair.records import canonical_hash
from tools.repair.shared_release import authenticate, bound, validate_completion

CAUSE = "RuntimeError: ValueError: Unregistered v3 inference feature channel: matcher_member"


def prior_costs(manifest):
    if "feature_boundary_recovery" not in manifest:
        return {}
    recovery = bound(manifest["feature_boundary_recovery"])
    original = bound(recovery["manifest"])
    original_identity = canonical_hash(original)
    # Pin a retained producer's own source separately from a corrected consumer.
    # Every other input-producer identity remains identical to the failed row.
    manifest = dict(manifest)
    if "public_input_producer" in manifest:
        contract = dict(manifest["public_input_producer"])
        expected_source = bound(contract["manifest"])["source_commit"]
        if contract.get("source_commit", expected_source) != expected_source:
            raise ValueError("Retained input producer source changed")
        contract.pop("source_commit", None)
        manifest["public_input_producer"] = contract
        original = dict(
            original,
            public_input_producer={
                k: v for k, v in original["public_input_producer"].items() if k != "source_commit"
            },
        )
    mutable = {"dispatch_nonce", "source_commit", "capacity_admission", "feature_boundary_recovery"}
    if recovery.get("schema") != "exact-repair/primary-feature-boundary-recovery/v1" or {
        k: v for k, v in original.items() if k not in mutable
    } != {k: v for k, v in manifest.items() if k not in mutable}:
        raise ValueError("Feature-boundary recovery changed scientific inputs or budgets")
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
        raise ValueError("Recovery lacks a terminal, retired original worker")
    work = Path(completion["work"])
    report = bound(dict(path=str(work / "report.json"), sha256=outputs["report.json"]))
    if report["manifest_identity"] != original_identity or [
        r["id"] for r in report["rows"]
    ] != [r["id"] for r in original["rows"]]:
        raise ValueError("Recovery report identities or denominator changed")
    attempted = [r for r in report["rows"] if r["status"] != "not_attempted"]
    if len(attempted) != 1:
        raise ValueError("Recovery requires exactly the diagnosed pre-result failure")
    row = attempted[0]
    declaration = next(r for r in original["rows"] if r["id"] == row["id"])
    relative = "rows/" + row["id"] + "/completion.json"
    saved = bound(dict(path=str(work / relative), sha256=outputs[relative]))
    spent = saved["charged_seconds"]
    if (
        saved != {k: v for k, v in row.items() if k != "id"}
        or saved["identity"] != canonical_hash((original_identity, declaration))
        or saved["status"] != "error"
        or saved.get("detail") != CAUSE
        or saved.get("cleanup_complete") is not True
        or saved.get("result") is not None
        or not math.isfinite(spent)
        or not 0 <= spent < declaration["case_seconds"] - 5
    ):
        raise ValueError("Recovery is not the bounded, cleaned-up feature-boundary failure")
    for artifact in saved["artifacts"]:
        authenticate(artifact)
    return {row["id"]: spent}
