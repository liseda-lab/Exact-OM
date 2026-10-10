"""Exact, separately authorized lower-cost calibration panel; no implicit quotas."""

from pathlib import Path
import math
import json
import hashlib
import time
from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha

LIMITS = dict(calibration=83, train=333, development=96, test=192)
FOUR_MODEL_LIMITS = dict(calibration=91, train=325, development=96, test=192)
PREVIOUS_LIMITS = dict(calibration=64, train=352, development=96, test=192)


def bound(reference):
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Teacher panel bound evidence changed")
    return read(reference["path"])


def contract(manifest):
    return {k: v for k, v in manifest.items() if k not in {"authorized", "teacher_panel"}}


def stage_elapsed(proposal):
    """Keep the original clock; admit a later window only with bound user approval."""
    stage = proposal["stage"]
    elapsed = 21600
    if proposal.get("timing_authorization"):
        approval = bound(proposal["timing_authorization"])
        previous = bound(approval["previous_proposal"])["stage"]
        approved, window = approval["approved_epoch"], approval["window_seconds"]
        if not (
            approval.get("schema") == "exact-repair/teacher-panel-timing-authorization/v1"
            and approval.get("status") == "approved"
            and approval.get("approval_text")
            and all(type(v) in (int, float) and math.isfinite(v) for v in (approved, window))
            and previous["deadline_epoch"] <= approved <= time.time()
            and 900 / 0.7 <= window <= 7200
            and stage["started_epoch"] == previous["started_epoch"]
            and stage["original_elapsed_seconds"] == previous["original_elapsed_seconds"]
            and stage["deadline_epoch"] == approved + window
        ):
            raise ValueError("Invalid approved teacher panel timing successor")
        elapsed = stage["deadline_epoch"] - stage["started_epoch"]
    if not (
        stage["original_elapsed_seconds"] == 14400
        and stage["proposed_elapsed_seconds"] == elapsed
        and stage["deadline_epoch"] == stage["started_epoch"] + elapsed
        and stage["latest_admission_epoch"] == stage["deadline_epoch"] - 900 / 0.7
    ):
        raise ValueError("Teacher panel cannot reset calibration time")
    return elapsed


def panel_contract(manifest, *, require_approval=True):
    ref = manifest["teacher_panel"]
    proposal = bound(ref["proposal"])
    prior = bound(proposal["previous_phase_state"])
    previous = bound(proposal["previous_manifest"])
    profile = manifest["profile"]
    if require_approval:
        approval = bound(ref["authorization"])
        if (
            approval.get("schema") != "exact-repair/teacher-panel-authorization/v1"
            or approval.get("status") != "approved"
            or not approval.get("approval_text")
            or approval.get("proposal") != ref["proposal"]
            or approval.get("timing_authorization") != proposal.get("timing_authorization")
        ):
            raise PermissionError("Teacher panel requires approval of the exact prepared amendment")
    four_models = proposal["max_requests"] == 32
    limits = FOUR_MODEL_LIMITS if four_models else LIMITS
    if not (
        proposal.get("schema") == "exact-repair/teacher-panel-proposal/v1"
        and manifest["phase"] == "calibration"
        and manifest["request_limits"] == proposal["request_limits"] == limits
        and prior["request_limits"] == previous["request_limits"] == PREVIOUS_LIMITS
        and manifest["request_budget_amendment"] == previous["request_budget_amendment"]
        and prior["lineage"] == previous["lineage_id"] == manifest["lineage_id"]
        and Path(manifest["ledger_directory"]).resolve()
        == Path(previous["ledger_directory"]).resolve()
        and manifest["cost_ceiling_usd"] == prior["cost_ceiling_usd"] == 35
        and proposal["max_requests"] in (24, 32)
        and 0 < proposal["max_reserved_usd"] <= (0.13296 if four_models else 0.11376)
        and proposal["calibration_comparison_identity_limit"] == (56 if four_models else 48)
        and len(prior["reservations"]) == proposal["prior_requests"] == 59
        and all(r["phase"] == "calibration" for r in prior["reservations"].values())
        and math.isclose(
            sum(r["reserved_cost_usd"] for r in prior["reservations"].values()),
            proposal["prior_reserved_usd"],
            rel_tol=0,
            abs_tol=1e-9,
        )
        and canonical_hash(contract(manifest)) == proposal["contracts"][profile]
        and len(manifest["slots"]) == 8
        and {r["profile"] for r in manifest["slots"]} == {profile}
        and manifest["seconds"] == 900
        and manifest["deadline_epoch"] == proposal["stage"]["deadline_epoch"]
    ):
        raise ValueError("Incompatible teacher panel contract or cumulative lineage")
    # All four authored packets, both orders. No outcome-driven substitutions.
    expected = {
        (p["path"], p["sha256"], order) for p in proposal["packets"] for order in (False, True)
    }
    actual = {(r["packet"]["path"], r["packet"]["sha256"], r["swapped"]) for r in manifest["slots"]}
    if actual != expected or len(expected) != 8 or len({r["id"] for r in manifest["slots"]}) != 8:
        raise ValueError("Teacher panel must retain all four packets and both orders")
    stage_elapsed(proposal)
    if profile == "glm":
        for row in manifest["slots"]:
            lineage = proposal["glm_recovery"][row["id"]]
            receipt = bound(lineage["receipt"])
            wire = bound(lineage["wire"])
            from exact.llm.ledger import request_identity

            raw = wire["raw_response"]
            response = json.loads(raw)
            old = prior["reservations"].get(canonical_hash(("calibration", lineage["old_slot"])))
            payload = wire["identity"]["payload"]
            context = json.loads(payload["messages"][1]["content"])
            if not (
                receipt["status"] == "annotation_unavailable"
                and request_identity(wire["identity"])[0] == wire["request_id"]
                and hashlib.sha256(raw.encode()).hexdigest() == wire["sha256"]
                and response["model"] == payload["model"] == manifest["profiles"][profile]["model"]
                and response.get("provider") in payload["provider"]["only"]
                and response["choices"][0]["finish_reason"] == "length"
                and old is not None
                and old["comparison_id"] == row["comparison_id"]
                and old["packet_hash"] == context["packet"]["context"]["packet_hash"]
                and context["swapped"] == row["swapped"]
                and lineage["same_cause_unsuccessful_attempts"] == 1
            ):
                raise ValueError("GLM recovery lacks its original definitive truncation lineage")
    return proposal, prior, previous


def reserve_panel(manifest, proposal, prior, records, slot, packet_hash, comparison_id, cost):
    """Called under the existing atomic phase-ledger lock before any transmission."""
    if any(records.get(k) != v for k, v in prior["reservations"].items()):
        raise ValueError("Teacher panel would erase or alter previous requests")
    rows = [r for r in manifest["slots"] if r["id"] == slot]
    if len(rows) != 1:
        raise ValueError("Unscheduled teacher panel slot")
    row = rows[0]
    packet = bound(row["packet"])
    from exact.repair.records import read_record

    if read_record(packet).content_hash != packet_hash or row["comparison_id"] != comparison_id:
        raise ValueError("Teacher panel comparison or packet changed")
    completed_panel = [r for r in records.values() if r.get("teacher_panel")]
    if not completed_panel and time.time() > proposal["stage"]["latest_admission_epoch"]:
        raise ValueError("Full teacher panel no longer fits 70 percent of remaining stage")
    if (
        len(completed_panel) >= proposal["max_requests"]
        or sum(r["reserved_cost_usd"] for r in completed_panel) + cost
        > proposal["max_reserved_usd"] + 1e-9
    ):
        raise ValueError("Teacher panel allowance exhausted")
    # The batch's cumulative stage ledger must carry the approved extension too.
    stage = read(proposal["stage"]["ledger_path"])
    actual = stage["stages"]["calibration"]
    if (
        actual["started_epoch"] != proposal["stage"]["started_epoch"]
        or stage["stage_limits"]["calibration"]["elapsed_seconds"] != stage_elapsed(proposal)
    ):
        raise ValueError("Approved stage amendment has not been applied without resetting time")
    if time.time() + 92 > proposal["stage"]["deadline_epoch"]:
        raise ValueError("Insufficient unchanged calibration stage balance")
    return dict(proposal=manifest["teacher_panel"]["proposal"], profile=manifest["profile"])
