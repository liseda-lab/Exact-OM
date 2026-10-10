"""Small, bounded semantic annotation runner shared by acquisition and decoded DEV.

Evidence and pair slots are frozen before annotation. The existing adapter owns
wire provenance; this module adds campaign/phase quotas and post-decode binding.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import (
    VerificationReportV2,
    canonical_hash,
    canonical_json,
    read_record,
)
from exact.repair.semantic_fidelity import (
    AGGREGATION_REVISION,
    EVALUATOR,
    FIDELITY_TRAINING_SCHEMA,
    PROMPT_VERSION,
    TEACHER,
    AnnotationBudget,
    AnnotationScheduleV3,
    SelectionAnnotationRunV1,
    ControlledSelectionAnnotationRunV1,
    SemanticAnnotationAdapter,
    SemanticConsequenceReportV3,
    SemanticEvidencePacketV3,
    SemanticPlanV3,
    ValidatedFidelityAggregateV3,
    aggregate_comparisons,
    annotation_prompt,
    consequence_basis_from_probes,
    semantic_plan_from_verification,
)
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import immutable

PHASE_LIMITS = {"calibration": 32, "train": 384, "development": 96, "test": 192}
SERVICE_LIMITS = {"train": 8 * 3600, "development": 4 * 3600, "test": 8 * 3600}
COMPARISON_LIMITS = {"calibration": 32, "train": 256, "development": 64, "test": 128}
REALLOCATED_LIMITS = {"calibration": 64, "train": 352, "development": 96, "test": 192}


class AnnotationBudgetExhausted(ValueError):
    """Expected terminal masking at a frozen cumulative quota, never a retry."""


class ConfirmedAnnotationFailure(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(
            f"Confirmed annotation provider failure: HTTP {evidence['http_status']}; "
            "durable attempt and cumulative reservations retained"
        )


def _confirmed_provider_failure(ledger, parameters):
    """Qualify a provider response against this exact request, without exposing credentials."""
    with ledger._transaction() as db:
        attempts = db.execute(
            "SELECT a.*,r.identity FROM attempts a JOIN requests r USING(request_id) "
            "WHERE a.status>=400 AND a.status<600"
        ).fetchall()
    for attempt in attempts:
        identity = json.loads(attempt["identity"])
        payload = identity.get("payload", {})
        if (
            identity.get("role") != parameters["role"]
            or payload.get("model") != parameters["model"]
            or canonical_hash(payload.get("messages")) != canonical_hash(parameters["messages"])
            or any(
                canonical_hash(payload.get(k)) != canonical_hash(parameters.get(k))
                for k in ("reasoning", "response_format", "max_tokens", "provider", "temperature")
            )
        ):
            continue
        raw = bytes(attempt["raw"] or b"")
        if not raw or hashlib.sha256(raw).hexdigest() != attempt["sha256"]:
            raise ValueError("Provider failure receipt has invalid response provenance")
        return dict(
            request_id=attempt["request_id"],
            attempt=attempt["number"],
            http_status=attempt["status"],
            response_sha256=attempt["sha256"],
            kind="authentication" if attempt["status"] in {401, 403} else "provider_response",
            retry_permitted=False,
            costs_reset=False,
        )
    return None


def _remaining_seconds(seconds, manifest):
    limits = [float(seconds)]
    for deadline in (manifest.get("deadline_epoch"), os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH")):
        if deadline is not None:
            value = float(deadline)
            if not math.isfinite(value):
                raise ValueError("Annotation deadline must be finite")
            limits.append(value - time.time())
    return max(0.0, min(limits))


def validate_manifest(value):
    if value.get("schema") != "exact-repair/corrective-annotation/v1":
        raise ValueError("Unknown corrective annotation manifest")
    if value.get("postprocessing_only"):
        raise PermissionError("Postprocessing contracts cannot transmit annotations")
    if not 0 < value["cost_ceiling_usd"] <= 35:
        raise ValueError("The authorized campaign ceiling is $35")
    if value["phase"] not in PHASE_LIMITS:
        raise ValueError("Unregistered annotation phase")
    _request_budget(value)
    _manifest_prompt(value)
    from tools.repair.annotation_profile import request_profile

    for name in value.get("request_profiles", {}):
        request_profile(value, name)
    if value.get("request_profiles") and not (
        value.get("teacher_panel") or value.get("qualified_teacher")
    ):
        raise PermissionError("Controlled profiles require their explicit teacher-panel amendment")
    if value["phase"] != "calibration":
        gate = value["calibration_gate"]
        if sha(gate["path"]) != gate["sha256"] or read(gate["path"])["status"] != "qualified":
            raise ValueError("Primary annotation requires a source-bound calibration gate")
    if value["phase"] == "development" and "frozen_annotation_slots" in value:
        slots = value["frozen_annotation_slots"]
        if len(slots) > COMPARISON_LIMITS["development"] or any(
            key != canonical_hash(row["selection_slot"])
            or type(row.get("swapped", False)) is not bool
            for key, row in slots.items()
        ):
            raise ValueError("Invalid frozen DEV comparison schedule")
        if sum(row.get("swapped", False) for row in slots.values()) < math.ceil(0.2 * len(slots)):
            raise ValueError(
                "DEV requires order-swap audits for at least 20% of frozen comparisons"
            )
    if not value.get("authorized"):
        raise PermissionError("Frozen annotation execution is not authorized")
    _provider_deferrals(value)
    _calibration_exclusions(value)
    if value.get("interface_recovery"):
        _interface_recovery(value)
    return value


def _manifest_prompt(manifest):
    """Legacy manifests retain their wire bytes; revisions require both bindings."""
    if "prompt_version" not in manifest and "prompt_hash" not in manifest:
        return PROMPT_VERSION
    version = manifest.get("prompt_version")
    prompt = annotation_prompt(version)
    if manifest.get("prompt_hash") != canonical_hash(prompt):
        raise ValueError("Frozen manifest prompt identity mismatch")
    return version


def _read_bound(reference):
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Frozen request amendment evidence changed")
    return read(reference["path"])


def _interface_recovery(manifest):
    """An explicit, finite approval is separate from the original 401 replacement.

    Preparing a proposal or leaving headroom in a request cap grants no new
    transmission authority. Bind every executable setting and retain all costs.
    """
    recovery = manifest["interface_recovery"]
    if not manifest.get("authorized") or not recovery.get("authorization"):
        raise PermissionError("Interface recovery requires its explicit scope amendment")
    proposal = _read_bound(recovery["proposal"])
    approval = _read_bound(recovery["authorization"])
    contract = {k: v for k, v in manifest.items() if k not in {"authorized", "interface_recovery"}}
    # run() specializes the default profile for each frozen slot.
    contract["profile"] = proposal["profile"]
    if not (
        proposal.get("schema") == "exact-repair/annotation-interface-recovery-proposal/v1"
        and approval.get("schema") == "exact-repair/annotation-interface-recovery-authorization/v1"
        and approval.get("status") == "approved"
        and approval.get("approval_text")
        and approval.get("proposal") == recovery["proposal"]
        and proposal["request_contract_hash"] == canonical_hash(contract)
        and manifest["phase"] == "calibration"
        and manifest["request_limits"] == REALLOCATED_LIMITS
        and manifest.get("prompt_version") == "semantic-fidelity-prompt/v3.2"
        and 0 < proposal["max_requests"] <= 13
        and 0 < proposal["max_reserved_usd"] <= 2
        and len(manifest["slots"]) == proposal["max_requests"]
        and len({row["id"] for row in manifest["slots"]}) == len(manifest["slots"])
        and set(proposal["slot_lineage"]) == {row["id"] for row in manifest["slots"]}
        and all(row["profile"] == proposal["profile"] for row in manifest["slots"])
    ):
        raise ValueError("Interface recovery differs from its approved finite contract")
    prior = _read_bound(proposal["previous_phase_state"])
    if (
        prior["lineage"] != manifest["lineage_id"]
        or prior["request_limits"] != REALLOCATED_LIMITS
        or prior["cost_ceiling_usd"] != manifest["cost_ceiling_usd"]
    ):
        raise ValueError("Interface recovery cannot change cumulative accounting")
    return proposal, prior


def _interface_recovery_proof(manifest, slot, packet_hash, comparison_id):
    from exact.repair.semantic_fidelity import _receipt_response, validate_comparison

    proposal, prior = _interface_recovery(manifest)
    rows = [row for row in manifest["slots"] if row["id"] == slot]
    if len(rows) != 1:
        raise ValueError("Interface recovery requires an exact scheduled slot")
    row = rows[0]
    lineage = proposal["slot_lineage"][slot]
    receipt = _read_bound(lineage["receipt"])
    schedule = read_record(_read_bound(lineage["schedule"]))
    wire = _read_bound(lineage["wire_receipt"])
    parameters = schedule.slots[0]["parameters"]
    response = _receipt_response(wire, parameters)
    old = prior["reservations"].get(canonical_hash(("calibration", lineage["slot_id"])))
    if (
        len(schedule.slots) != 1
        or receipt["status"] != "annotation_unavailable"
        or old is None
        or old["packet_hash"] != packet_hash
        or schedule.packet.content_hash != packet_hash
        or schedule.packet.split != "train"
        or old.get("comparison_id", old["slot"]) != comparison_id
        or row["comparison_id"] != comparison_id
        or manifest["profile"] != row["profile"]
        or parameters["model"] != manifest["profiles"][row["profile"]]["model"]
        or canonical_hash(parameters["provider"])
        != canonical_hash(manifest["profiles"][row["profile"]]["provider"])
        or json.loads(parameters["messages"][1]["content"])["swapped"] != row["swapped"]
        or response.get("model") != parameters["model"]
        or response.get("provider") not in parameters["provider"].get("only", [])
        or response.get("choices", [{}])[0].get("finish_reason") != "stop"
        or response["choices"][0].get("message", {}).get("refusal")
    ):
        raise ValueError("Interface recovery changed its failed request lineage")
    from exact.llm.routing import extract_chat_text

    try:
        validate_comparison(extract_chat_text(response), schedule.packet, swapped=row["swapped"])
    except ValueError:
        pass
    else:
        raise ValueError("A valid scientific judgment is not an interface failure to replay")
    return (
        proposal,
        prior,
        dict(
            proposal_sha256=manifest["interface_recovery"]["proposal"]["sha256"],
            parent_request_id=wire["request_id"],
            parent_slot=lineage["slot_id"],
            cause="annotation-request-interface/v3.1",
            technical_attempt=2,
        ),
    )


def _provider_policy_failure(manifest, evidence):
    """Only a qualified account training-policy rejection disqualifies one provider."""
    if evidence.get("http_status") != 404 or evidence.get("kind") != "provider_response":
        return False
    path = Path(manifest["ledger_directory"]) / "requests.sqlite3"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT status,sha256,raw FROM attempts WHERE request_id=? AND number=?",
            (evidence["request_id"], evidence["attempt"]),
        ).fetchone()
    if row is None or row[0] != 404 or row[1] != evidence["response_sha256"]:
        raise ValueError("Provider-policy failure changed its durable attempt")
    raw = bytes(row[2] or b"")
    if hashlib.sha256(raw).hexdigest() != row[1]:
        raise ValueError("Provider-policy failure changed its response digest")
    error = json.loads(raw).get("error", {})
    metadata = error.get("metadata", {})
    reasons = metadata.get("ineligibility_reasons", [])
    return (
        error.get("code") == 404
        and metadata.get("failed_routing_step") == "Filter by Guardrails"
        and metadata.get("input_endpoint_count", 0) > 0
        and len(reasons) == 1
        and reasons[0].get("reason") == "paid-model-training-violation-by-account"
        and reasons[0].get("endpoint_count") == metadata["input_endpoint_count"]
        and reasons[0].get("configure_url") == "https://openrouter.ai/settings/privacy"
    )


def _provider_deferrals(manifest):
    """A successor may mask a failed profile, never alter its frozen comparisons."""
    from exact.llm.ledger import RequestLedger

    result = {}
    for profile, binding in manifest.get("provider_deferrals", {}).items():
        previous = _read_bound(binding["previous_manifest"])
        unchanged = lambda value: {
            key: item
            for key, item in value.items()
            if key not in {"provider_deferrals", "calibration_exclusions", "profile"}
        }
        if (
            manifest["phase"] != "calibration"
            or unchanged(manifest) != unchanged(previous)
            or profile not in previous["profiles"]
        ):
            raise ValueError("Provider deferral changed the frozen calibration design")
        matches = [row for row in previous["slots"] if row["id"] == binding["failed_slot"]]
        if len(matches) != 1 or matches[0].get("profile", previous["profile"]) != profile:
            raise ValueError("Provider deferral does not identify its failed profile")
        row = matches[0]
        receipt = _read_bound(binding["receipt"])
        directory = Path(binding["receipt"]["path"]).parent
        packet = read_record(read(directory / "packet.json"))
        schedule = read_record(read(directory / "schedule.json"))
        expected = canonical_hash(
            (
                packet,
                {**previous, "profile": profile},
                row["id"],
                row.get("swapped", False),
                row.get("comparison_id", row["id"]),
            )
        )
        if (
            receipt.get("status") != "provider_error"
            or receipt["identity"] != expected
            or sha(row["packet"]["path"]) != row["packet"]["sha256"]
            or packet != read_record(read(row["packet"]["path"]))
            or schedule.packet != packet
        ):
            raise ValueError("Provider deferral changed its exact failed-request provenance")
        evidence = _confirmed_provider_failure(
            RequestLedger(Path(manifest["ledger_directory"])), schedule.slots[0]["parameters"]
        )
        if (
            evidence is None
            or evidence != receipt.get("confirmed_failure")
            or not _provider_policy_failure(manifest, evidence)
        ):
            raise ValueError("Provider deferral requires a qualified account-policy rejection")
        result[profile] = dict(failed_slot=row["id"], confirmed_failure=evidence, binding=binding)
    return result


def _calibration_exclusions(manifest):
    """Retain a terminal truncated profile and continue only untouched profiles.

    This is an explicit, receipt-bound successor, never a provider-error retry
    or a change to the original 8/8 qualification rule or comparison denominator.
    """
    from exact.llm.ledger import RequestLedger

    result = {}
    for profile, binding in manifest.get("calibration_exclusions", {}).items():
        previous = _read_bound(binding["previous_manifest"])
        unchanged = lambda value: {
            key: item
            for key, item in value.items()
            if key not in {"calibration_exclusions", "profile"}
        }
        if (
            manifest["phase"] != "calibration"
            or unchanged(manifest) != unchanged(previous)
            or previous.get("selection_rule")
            != (
                "Require8/8valid grounded judgments,8/8declared semantic decisions,"
                "4/4order consistency;lowest frozen maximum cost among eligible cheap models;"
                "Sonnet fallback onlyifqualified"
            )
            or profile in manifest.get("provider_deferrals", {})
        ):
            raise ValueError("Calibration exclusion changed the frozen qualification design")
        report = _read_bound(binding["previous_report"])
        slots = {row["id"]: row for row in previous["slots"]}
        reported = {row["id"]: row for row in report["rows"]}
        profile_slots = {
            key: row
            for key, row in slots.items()
            if row.get("profile", previous["profile"]) == profile
        }
        if (
            len(profile_slots) != 8
            or len(reported) != len(report["rows"])
            or set(reported) != set(slots)
            or report.get("scheduled") != len(slots)
        ):
            raise ValueError("Calibration exclusion lost scheduled rows")
        if any(
            reported[key]["status"] != "not_attempted_provider_failure"
            for key, row in slots.items()
            if row.get("profile", previous["profile"]) != profile
            and row.get("profile", previous["profile"])
            not in manifest.get("provider_deferrals", {})
        ):
            raise ValueError("Calibration continuation may only admit untouched other profiles")
        attempted = {
            key
            for key in profile_slots
            if reported[key]["status"] in {"unavailable", "provider_error"}
        }
        if (
            set(binding["receipts"]) != attempted
            or binding["failed_gate_slot"] not in attempted
            or any(
                reported[key]["status"]
                not in {"unavailable", "provider_error", "not_attempted_provider_failure"}
                for key in profile_slots
            )
        ):
            raise ValueError("Calibration exclusion must retain every attempted slot")
        retained = {}
        ledger = RequestLedger(Path(manifest["ledger_directory"]))
        for key in sorted(attempted):
            row = profile_slots[key]
            reference = binding["receipts"][key]
            receipt = _read_bound(reference)
            directory = Path(reference["path"]).parent
            packet = read_record(read(directory / "packet.json"))
            schedule = read_record(read(directory / "schedule.json"))
            expected = canonical_hash(
                (
                    packet,
                    {**previous, "profile": profile},
                    key,
                    row.get("swapped", False),
                    row.get("comparison_id", key),
                )
            )
            if (
                receipt.get("identity") != expected
                or sha(row["packet"]["path"]) != row["packet"]["sha256"]
                or packet != read_record(read(row["packet"]["path"]))
                or schedule.packet != packet
                or reported[key].get("artifact") is not None
            ):
                raise ValueError("Calibration exclusion changed attempted-slot provenance")
            parameters = schedule.slots[0]["parameters"]
            if receipt.get("status") == "provider_error":
                failure = _confirmed_provider_failure(ledger, parameters)
                if (
                    failure is None
                    or failure != reported[key].get("confirmed_failure")
                    or failure != receipt.get("confirmed_failure")
                ):
                    raise ValueError("Calibration exclusion requires a definitive provider receipt")
            elif (
                receipt.get("status") != "annotation_unavailable"
                or reported[key]["status"] != "unavailable"
            ):
                raise ValueError(
                    "Calibration exclusion cannot mask unknown delivery or usable labels"
                )
            if key == binding["failed_gate_slot"]:
                with ledger._transaction() as db:
                    attempts = db.execute(
                        "SELECT a.*,r.identity FROM attempts a JOIN requests r USING(request_id) "
                        "WHERE a.status=200"
                    ).fetchall()
                truncated = False
                for attempt in attempts:
                    identity = json.loads(attempt["identity"])
                    payload = identity.get("payload", {})
                    if identity.get("role") != parameters["role"] or any(
                        canonical_hash(payload.get(field)) != canonical_hash(parameters[field])
                        for field in ("model", "messages")
                    ):
                        continue
                    raw = bytes(attempt["raw"] or b"")
                    if hashlib.sha256(raw).hexdigest() != attempt["sha256"]:
                        raise ValueError("Calibration exclusion changed wire response digest")
                    choices = json.loads(raw).get("choices", [])
                    truncated = len(choices) == 1 and choices[0].get("finish_reason") == "length"
                if receipt.get("status") != "annotation_unavailable" or not truncated:
                    raise ValueError(
                        "Calibration exclusion requires a definitive truncated response"
                    )
            retained[key] = {**reported[key], "retained_receipt": reference}
        result[profile] = dict(binding=binding, retained_rows=retained)
    return result


def _request_budget(manifest):
    if manifest.get("qualified_teacher"):
        from tools.repair.teacher_selection import qualified_contract

        proposal, phase, source, previous = qualified_contract(manifest)
        _, amendment = _request_budget(previous)
        return proposal["request_limits"], {
            **amendment,
            "qualified_phase": phase,
            "qualified_panel_binding": source["teacher_panel"],
        }
    if manifest.get("teacher_panel"):
        from tools.repair.teacher_panel import panel_contract

        proposal, prior, previous = panel_contract(manifest)
        _, amendment = _request_budget(previous)
        return proposal["request_limits"], {**amendment, "panel": proposal, "panel_prior": prior}
    binding = manifest.get("request_budget_amendment")
    if binding is None:
        if manifest["request_limits"] != PHASE_LIMITS:
            raise ValueError("Request quotas require an explicit protocol amendment")
        return PHASE_LIMITS, None
    approval = _read_bound(binding["authorization"])
    proposal = _read_bound(approval["proposal"])
    prior = _read_bound(binding["previous_phase_state"])
    previous_manifest = _read_bound(binding["previous_manifest"])
    if not (
        approval.get("schema") == "exact-repair/request-budget-amendment-authorization/v1"
        and approval.get("status") == "approved"
        and approval.get("approval_text")
        and approval.get("original_request_limits")
        == proposal.get("original_request_limits")
        == PHASE_LIMITS
        and approval.get("proposed_request_limits")
        == proposal.get("proposed_request_limits")
        == manifest["request_limits"]
        == REALLOCATED_LIMITS
        and approval.get("aggregate_max_requests") == 704
        and approval.get("calibration_max_usd") == 2
        and approval.get("campaign_max_usd")
        == manifest["cost_ceiling_usd"]
        == prior["cost_ceiling_usd"]
        == 35
        and approval.get("replacement_calibration_max_requests") == 32
        and prior["request_limits"] == previous_manifest["request_limits"] == PHASE_LIMITS
        and prior["lineage"] == previous_manifest["lineage_id"] == manifest["lineage_id"]
        and Path(previous_manifest["ledger_directory"]).resolve()
        == Path(manifest["ledger_directory"]).resolve()
        and len(prior["reservations"]) == 32
        and all(row["phase"] == "calibration" for row in prior["reservations"].values())
        and math.isclose(
            sum(row["reserved_cost_usd"] for row in prior["reservations"].values()),
            approval["prior_reserved_exposure_usd"],
            rel_tol=0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Unsupported or incompatible request-budget amendment")
    return REALLOCATED_LIMITS, dict(
        binding=binding, approval=approval, prior=prior, previous_manifest=previous_manifest
    )


def _replacement_proof(manifest, slot, packet_hash, comparison_id, amendment):
    """One successor for one exact rejected request; never an unknown-delivery retry."""
    from exact.llm.ledger import RequestLedger

    rows = [row for row in manifest["slots"] if row["id"] == slot]
    if len(rows) != 1 or "replaces" not in rows[0]:
        raise ValueError("Amended calibration requires one frozen replacement slot")
    replacement = rows[0]["replaces"]
    old_slot = replacement["slot_id"]
    prior = amendment["prior"]["reservations"].get(canonical_hash(("calibration", old_slot)))
    old_rows = [row for row in amendment["previous_manifest"]["slots"] if row["id"] == old_slot]
    if (
        slot == old_slot
        or prior is None
        or len(old_rows) != 1
        or prior["packet_hash"] != packet_hash
        or comparison_id != prior.get("comparison_id", old_slot)
    ):
        raise ValueError("Replacement changed original comparison or packet")
    old_row = old_rows[0]
    old_manifest = amendment["previous_manifest"]
    old_profile = old_row.get("profile", old_manifest["profile"])
    new_profile = rows[0].get("profile", manifest["profile"])
    public = lambda profile: {
        key: value for key, value in profile.items() if key not in {"api_key_path", "api_key_env"}
    }
    if public(old_manifest["profiles"][old_profile]) != public(
        manifest["profiles"][new_profile]
    ) or old_row.get("swapped", False) != rows[0].get("swapped", False):
        raise ValueError("Replacement changed frozen model/provider/order")
    receipt = _read_bound(replacement["receipt"])
    directory = Path(replacement["receipt"]["path"]).parent
    packet = read_record(read(directory / "packet.json"))
    expected_identity = canonical_hash(
        (
            packet,
            {**old_manifest, "profile": old_profile},
            old_slot,
            old_row.get("swapped", False),
            old_row.get("comparison_id", old_slot),
        )
    )
    schedule = read_record(read(directory / "schedule.json"))
    if (
        receipt["identity"] != expected_identity
        or packet.content_hash != packet_hash
        or schedule.packet != packet
    ):
        raise ValueError("Replacement failed-request provenance changed")
    evidence = _confirmed_provider_failure(
        RequestLedger(Path(manifest["ledger_directory"])), schedule.slots[0]["parameters"]
    )
    if evidence is None or evidence["http_status"] != 401:
        raise ValueError("Replacement requires the original definitive HTTP401 receipt")
    return dict(
        original_slot=old_slot,
        failed_attempt=evidence,
        authorization_sha256=amendment["binding"]["authorization"]["sha256"],
    )


def _failure_report_fields(evidence):
    authentication = evidence["kind"] == "authentication"
    return dict(
        status="blocked_external" if authentication else "failed",
        confirmed_failure=evidence,
        **(
            dict(
                blocker=dict(
                    kind="provider_authentication",
                    status_code=evidence["http_status"],
                    retry_permitted=False,
                    requires_user=True,
                    detail="OpenRouter rejected the configured credential; update the authorized credential before an explicitly admitted resume.",
                )
            )
            if authentication
            else {}
        ),
    )


def _authentication_preflight(manifest, output):
    """Free read-only credential check; no generation or scientific reservation."""
    from exact.llm.routing import LLMRouter

    router = LLMRouter(manifest["profiles"])
    seen = set()
    checked = []
    try:
        for name in dict.fromkeys(
            row.get("profile", manifest["profile"]) for row in manifest["slots"]
        ):
            if name in manifest.get("provider_deferrals", {}) or name in manifest.get(
                "calibration_exclusions", {}
            ):
                continue
            profile = router.profiles[name]
            key = router.hosted.resolve_api_key(profile)
            if not key:
                raise RuntimeError("Annotation credential unavailable before any transmission")
            if (profile.api_base, key) in seen:
                continue
            seen.add((profile.api_base, key))  # In-memory only; never retained in artifacts.
            left = _remaining_seconds(manifest["seconds"], manifest)
            if left <= 2:
                return None
            response = router.hosted._client.request(
                method="GET",
                url=profile.api_base + "/key",
                headers={"Authorization": "Bearer " + key},
                timeout=min(10.0, left - 2),
            )
            receipt = dict(
                profile=name,
                method="GET",
                endpoint=profile.api_base + "/key",
                http_status=response.status_code,
                generation_requests=0,
                response_sha256=hashlib.sha256(response.content).hexdigest(),
            )
            if response.is_error:
                evidence = dict(receipt, raw_response=response.content.decode("utf-8"))
                evidence_path = (
                    Path(output) / "authentication-evidence" / (canonical_hash(evidence) + ".json")
                )
                immutable(evidence_path, evidence)
                receipt["evidence"] = dict(
                    path=str(evidence_path.resolve()), sha256=sha(evidence_path)
                )
            checked.append(receipt)
            write_artifact(
                Path(output) / "authentication.json",
                dict(schema="exact-repair/annotation-authentication/v1", checks=checked),
            )
            if response.is_error:
                return dict(
                    **receipt,
                    kind=(
                        "authentication"
                        if response.status_code in {401, 403}
                        else "provider_response"
                    ),
                    retry_permitted=False,
                    costs_reset=False,
                    source="read_only_authentication_preflight",
                )
        return None
    finally:
        router.hosted.close()


def _reserve_phase(manifest, slot, packet_hash, cost, *, comparison_id=None):
    """Fail closed after a killed sender; replacement jobs never replenish quotas."""
    path = Path(manifest["ledger_directory"]) / "phase-reservations.json"
    limits, amendment = _request_budget(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            read(path)
            if path.exists()
            else dict(
                schema="exact-repair/annotation-phase-budget/v1",
                lineage=manifest["lineage_id"],
                cost_ceiling_usd=manifest["cost_ceiling_usd"],
                request_limits=PHASE_LIMITS,
                reservations={},
            )
        )
        if (state["lineage"], state["cost_ceiling_usd"]) != (
            manifest["lineage_id"],
            manifest["cost_ceiling_usd"],
        ):
            raise ValueError("Annotation budget lineage changed")
        if amendment:
            binding = amendment["binding"]
            if state["request_limits"] == PHASE_LIMITS:
                if canonical_hash(state) != canonical_hash(amendment["prior"]):
                    raise ValueError("Original request ledger changed before amendment")
                state["request_limits"] = limits
                state["request_budget_amendment"] = binding
            if state.get("request_budget_amendment") != binding or any(
                state["reservations"].get(key) != value
                for key, value in amendment["prior"]["reservations"].items()
            ):
                raise ValueError("Request amendment would reset or modify original attempts")
        if amendment and amendment.get("panel"):
            prior = amendment["panel_prior"]
            if state["request_limits"] == prior["request_limits"]:
                if canonical_hash(state) != canonical_hash(prior):
                    raise ValueError("Teacher panel prior ledger changed before amendment")
                state["request_limits"] = limits
                state["teacher_panel_amendment"] = manifest["teacher_panel"]
            if state.get("teacher_panel_amendment") != manifest["teacher_panel"]:
                raise ValueError("Teacher panel amendment binding changed")
        if amendment and amendment.get("qualified_phase"):
            prior = amendment["qualified_phase"]
            if state.get("teacher_panel_amendment") != amendment["qualified_panel_binding"] or any(
                state["reservations"].get(k) != v for k, v in prior["reservations"].items()
            ):
                raise ValueError("Qualified teacher successor would reset calibration history")
        if state["request_limits"] != limits:
            raise ValueError("Annotation request limits changed without compatible amendment")
        identity = canonical_hash((manifest["phase"], slot))
        records = state["reservations"]
        previous = records.get(identity)
        if previous:
            if previous["packet_hash"] != packet_hash or previous.get(
                "request_basis"
            ) != canonical_hash((manifest, slot, packet_hash)):
                raise ValueError("Frozen annotation slot changed its evidence/plan")
            return previous["state"] == "completed"
        same = [row for row in records.values() if row["phase"] == manifest["phase"]]
        comparison_id = comparison_id or slot
        replacement = None
        interface_recovery = None
        teacher_panel = None
        if amendment and amendment.get("panel"):
            from tools.repair.teacher_panel import reserve_panel

            teacher_panel = reserve_panel(
                manifest,
                amendment["panel"],
                amendment["panel_prior"],
                records,
                slot,
                packet_hash,
                comparison_id,
                cost,
            )
        elif manifest.get("interface_recovery"):
            proposal, prior, interface_recovery = _interface_recovery_proof(
                manifest, slot, packet_hash, comparison_id
            )
            if any(records.get(key) != value for key, value in prior["reservations"].items()):
                raise ValueError("Interface recovery would reset earlier attempts")
            recovered = [row for row in records.values() if row.get("interface_recovery")]
            if (
                not recovered
                and time.time() > proposal["time"]["latest_full_panel_admission_epoch"]
            ):
                raise AnnotationBudgetExhausted("Complete interface panel no longer fits stage")
            if any(
                row["interface_recovery"]["parent_request_id"]
                == interface_recovery["parent_request_id"]
                for row in recovered
            ):
                raise ValueError("Interface failure already has its one technical retry")
            if (
                len(recovered) >= proposal["max_requests"]
                or sum(row["reserved_cost_usd"] for row in recovered) + cost
                > proposal["max_reserved_usd"] + 1e-9
            ):
                raise AnnotationBudgetExhausted("Interface recovery allowance exhausted")
        elif amendment and manifest["phase"] == "calibration":
            replacement = _replacement_proof(manifest, slot, packet_hash, comparison_id, amendment)
            if any(
                row.get("replacement", {}).get("original_slot") == replacement["original_slot"]
                for row in records.values()
            ):
                raise ValueError("Original rejected request already has its admitted replacement")
            spent = sum(
                row["reserved_cost_usd"] for row in records.values() if row.get("replacement")
            )
            if (
                spent + cost
                > amendment["approval"]["replacement_calibration_max_reserved_usd"] + 1e-9
            ):
                raise AnnotationBudgetExhausted("Replacement calibration exposure exhausted")
        unique = {row.get("comparison_id", row["slot"]) for row in same}
        comparison_limit = (
            amendment["panel"]["calibration_comparison_identity_limit"]
            if teacher_panel
            else COMPARISON_LIMITS[manifest["phase"]]
        )
        if comparison_id not in unique and len(unique) >= comparison_limit:
            raise AnnotationBudgetExhausted("Cumulative unique comparison quota exhausted")
        service_limit = SERVICE_LIMITS.get(manifest["phase"])
        # Charge the full frozen request allowance, including unknown delivery.
        # Old records have the same 90-second cap and are never replenished.
        if service_limit is not None and 90 * (len(same) + 1) > service_limit:
            raise AnnotationBudgetExhausted("Cumulative hosted service allowance exhausted")
        if len(same) >= limits[manifest["phase"]] or len(records) >= 704:
            raise AnnotationBudgetExhausted("Cumulative annotation request quota exhausted")
        if (
            sum(row["reserved_cost_usd"] for row in records.values()) + cost
            > state["cost_ceiling_usd"]
        ):
            raise AnnotationBudgetExhausted("Cumulative annotation monetary allowance exhausted")
        if (
            manifest["phase"] == "calibration"
            and sum(r["reserved_cost_usd"] for r in same) + cost > 2
        ):
            raise AnnotationBudgetExhausted("Calibration $2 allowance exhausted")
        records[identity] = dict(
            phase=manifest["phase"],
            slot=slot,
            packet_hash=packet_hash,
            reserved_cost_usd=cost,
            state="reserved",
            admitted_epoch=time.time(),
            comparison_id=comparison_id,
            request_basis=canonical_hash((manifest, slot, packet_hash)),
            **(dict(teacher_panel=teacher_panel) if teacher_panel else {}),
            **(dict(replacement=replacement) if replacement else {}),
            **(dict(interface_recovery=interface_recovery) if interface_recovery else {}),
        )
        write_artifact(path, state)
        return None  # The only state that permits a new transmission.


def _settle_phase(manifest, slot, *, completed):
    path = Path(manifest["ledger_directory"]) / "phase-reservations.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(path)
        state["reservations"][canonical_hash((manifest["phase"], slot))]["state"] = (
            "completed" if completed else "unresolved"
        )
        write_artifact(path, state)


def annotate_packet(
    packet, manifest, output, *, slot_id, swapped=False, seconds=94, comparison_id=None
):
    """One predeclared comparison, zero hidden retries, committed raw provenance."""
    from exact.llm.routing import LLMRouter

    started = time.monotonic()
    validate_manifest(manifest)
    output = Path(output)
    if swapped and comparison_id is None:
        raise ValueError("Order-swap audit requires its frozen original comparison identity")
    identity = canonical_hash((packet, manifest, slot_id, swapped, comparison_id or slot_id))
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["identity"] != identity:
            raise ValueError("Annotation resume dependencies changed")
        artifact = output / "labels.json"
        if receipt["status"] == "complete" and sha(artifact) != receipt["labels_sha256"]:
            raise ValueError("Committed annotation artifact changed")
        if receipt["status"] in {"provider_error", "annotation_unavailable"}:
            # Older workers masked HTTP failures. Revalidate their retained
            # wire receipt before permitting any later scheduled request.
            from exact.llm.ledger import RequestLedger

            schedule = read_record(read(output / "schedule.json"))
            failure = _confirmed_provider_failure(
                RequestLedger(Path(manifest["ledger_directory"])), schedule.slots[0]["parameters"]
            )
            if failure is not None:
                raise ConfirmedAnnotationFailure(failure)
        return artifact if receipt["status"] == "complete" else None
    # The request configuration is frozen across restart. A shorter remaining
    # slice is an unavailable slot, never a changed identity or hidden retry.
    if _remaining_seconds(seconds, manifest) < 92 or not packet.eligible:
        return None
    profile_name = manifest["profile"]
    prices = manifest["prices_per_million"][profile_name]
    # Conservative maximum includes billed reasoning tokens in the output cap.
    from tools.repair.annotation_profile import request_profile
    from exact.repair.annotation_controls import response_format, input_token_bound

    controls = request_profile(manifest, profile_name)
    output_cap = controls["max_output_tokens"] if controls else 2000
    cost = (8000 * prices["input"] + output_cap * prices["output"]) / 1_000_000
    if cost <= 0:
        raise ValueError("Finite positive frozen provider prices required")
    role = TEACHER if packet.split == "train" else EVALUATOR
    teacher = manifest["teacher_profile"]
    test = manifest["test_profile"]
    evaluator = profile_name if packet.split != "train" else test
    selected_teacher = profile_name if packet.split == "train" else teacher
    budgets = {
        TEACHER: AnnotationBudget(416, 416 * 10000, 35, 416 * 90),
        EVALUATOR: AnnotationBudget(288, 288 * 10000, 35, 288 * 90),
    }
    parent_splits = manifest["parent_splits"]
    if parent_splits.get(packet.parent_group_id) != packet.split:
        raise ValueError("Decoded packet changed frozen parent split")
    run_type = ControlledSelectionAnnotationRunV1 if controls else SelectionAnnotationRunV1
    run = run_type(
        **(
            dict(
                input_tokenizer=controls["tokenizer"],
                role_request_controls={
                    role: dict(
                        **(
                            {"reasoning": controls["reasoning"]}
                            if controls["reasoning"] is not None
                            else {}
                        ),
                        response_format=response_format(packet, swapped),
                    )
                },
            )
            if controls
            else {}
        ),
        run_id=canonical_hash((identity, "run")),
        lineage_id=manifest["lineage_id"],
        authorized=True,
        role_profiles={TEACHER: selected_teacher, EVALUATOR: evaluator},
        role_budgets=budgets,
        aggregate_budget=AnnotationBudget(704, 704 * 10000, manifest["cost_ceiling_usd"], 704 * 90),
        max_input_bytes=8000,
        max_output_tokens=output_cap,
        max_cost_per_request_usd=cost,
        max_seconds_per_request=90,
        comparisons_per_case=1000,
        repetitions=1,
        correction_cap=0,
        concurrency=4,
        independent_evaluator=packet.split == "test",
        rubric_version=packet.rubric_version,
        evidence_manifest_hash=canonical_hash((packet.content_hash,)),
        packet_hashes=(packet.content_hash,),
        split_manifest_hash=canonical_hash(parent_splits),
        parent_splits=parent_splits,
        data_permissions=manifest["data_permissions"],
        retention="campaign durable raw receipts",
        aggregation_rule=AGGREGATION_REVISION,
        prompt_version=_manifest_prompt(manifest),
        evaluator_split="test" if packet.split == "train" else packet.split,
        development_use_policy=(
            "development_selection/v1"
            if packet.split == "development"
            else "independent_evaluation"
        ),
        selection_model_ids=tuple(manifest["selection_model_ids"]),
    )
    router = LLMRouter(manifest["profiles"])
    adapter = SemanticAnnotationAdapter(router, run, Path(manifest["ledger_directory"]))
    schedule = adapter.schedule(packet, role=role, quorum=1, presentation_orders=(swapped,))
    # Bound the ENTIRE message, not just evidence, before spending. UTF-8 bytes
    # plus framing is conservative for all shortlisted text tokenizers.
    messages = schedule.slots[0]["parameters"]["messages"]
    if (
        input_token_bound(
            messages,
            (
                {
                    k: schedule.slots[0]["parameters"][k]
                    for k in ("reasoning", "response_format")
                    if k in schedule.slots[0]["parameters"]
                }
                if controls
                else None
            ),
            controls["tokenizer"] if controls else None,
        )
        > 8000
    ):
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="input_token_bound",
                retry_permitted=False,
                costs_reset=False,
            ),
        )
        router.hosted.close()
        return None
    if _remaining_seconds(seconds - (time.monotonic() - started), manifest) < 92:
        router.hosted.close()
        return None
    try:
        reserve = _reserve_phase(
            manifest, slot_id, packet.content_hash, cost, comparison_id=comparison_id
        )
    except AnnotationBudgetExhausted as error:
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="budget_unavailable",
                detail=str(error),
                retry_permitted=False,
                costs_reset=False,
            ),
        )
        router.hosted.close()
        return None
    if reserve is False:
        failure = _confirmed_provider_failure(adapter.ledger, schedule.slots[0]["parameters"])
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="provider_error" if failure else "unknown_delivery",
                confirmed_failure=failure,
                costs_reset=False,
                retry_permitted=False,
            ),
        )
        router.hosted.close()
        if failure:
            raise ConfirmedAnnotationFailure(failure)
        return None
    immutable(output / "packet.json", packet.to_dict())
    immutable(output / "run.json", run.to_dict())
    immutable(output / "schedule.json", schedule.to_dict())
    try:
        comparison = adapter.annotate(packet, role=role, swapped=swapped)
        aggregate = ValidatedFidelityAggregateV3(
            schedule, (comparison,), aggregate_comparisons((comparison,), schedule=schedule)
        )
        artifact = output / "labels.json"
        write_artifact(
            artifact,
            dict(
                schema=FIDELITY_TRAINING_SCHEMA,
                aggregation_revision=AGGREGATION_REVISION,
                comparisons=[dict(packet=packet.to_dict(), comparison=aggregate.to_dict())],
            ),
        )
        _settle_phase(manifest, slot_id, completed=True)
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="complete",
                labels_sha256=sha(artifact),
                eligible=aggregate.global_target_eligible,
                costs=adapter.summary(),
                costs_reset=False,
            ),
        )
        return artifact
    except (ValueError, RuntimeError) as error:
        _settle_phase(manifest, slot_id, completed=False)
        failure = _confirmed_provider_failure(adapter.ledger, schedule.slots[0]["parameters"])
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="provider_error" if failure else "annotation_unavailable",
                error=str(error),
                confirmed_failure=failure,
                costs=adapter.summary(),
                retry_permitted=False,
                costs_reset=False,
            ),
        )
        if failure is not None:
            raise ConfirmedAnnotationFailure(failure) from error
        return None
    finally:
        router.hosted.close()


def _verified_plan(case_record, assignment):
    from exact.repair.kernel import materialize, verify_assignment
    from exact.repair.learning import OwlTeacherOracle
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms
    from tools.repair.prepare import case_from_dict

    case = case_from_dict(case_record)
    report = verify_assignment(case.problem, tuple(assignment))
    if not report.authorizes:
        return None
    axioms, _ = materialize(case.problem, tuple(assignment))
    # Ask non-vacuity for every declared consequence, including unwanted ones.
    probes = tuple(dataclasses.replace(p, desired=True) for p in case.probes)
    oracle = OwlTeacherOracle(
        OwlVerifier("auto", backend="auto"), snapshot_from_axioms(axioms), probes
    )
    basis = consequence_basis_from_probes(case.probes)
    outcomes = {}
    for probe in probes:
        entailed = oracle.entails(probe.axiom)
        conditions = [oracle.satisfiable(condition) for condition in probe.conditions()]
        nonvacuity = (
            "not_applicable"
            if not conditions
            else "unknown" if None in conditions else "pass" if all(conditions) else "fail"
        )
        outcomes[probe.probe_id] = dict(
            status="unknown" if entailed is None else str(entailed).lower(),
            complete=entailed is not None and nonvacuity != "unknown",
            nonvacuity=nonvacuity,
        )
    consequence = SemanticConsequenceReportV3(
        report.theory_hash,
        report.policy_hash,
        canonical_hash(basis),
        outcomes,
        "capability-before-cost/installed-qualified-routes",
    )
    plan = semantic_plan_from_verification(
        case.problem,
        tuple(assignment),
        report,
        consequence_basis=basis,
        consequence_report=consequence,
    )
    return plan


def annotate_comparison(
    packet, manifest, output: str | Path, *, comparison_id, order_swap_audit, seconds
) -> Path | None:
    """Original plus a scheduled audit; missing or discordant audits cannot label."""
    started = time.monotonic()
    output = Path(output)
    original: Path | None = annotate_packet(
        packet,
        manifest,
        output / "original",
        slot_id=comparison_id + ":original",
        comparison_id=comparison_id,
        seconds=seconds,
    )
    if original is None or not order_swap_audit:
        return original
    swapped: Path | None = annotate_packet(
        packet,
        manifest,
        output / "swapped",
        slot_id=comparison_id + ":swapped",
        comparison_id=comparison_id,
        swapped=True,
        seconds=seconds - (time.monotonic() - started),
    )
    if swapped is None:
        write_artifact(
            output / "audit.json",
            dict(
                status="required_swap_unavailable",
                comparison_id=comparison_id,
                original_labels_sha256=sha(original),
                scheduled=2,
                available=1,
                global_target_eligible=False,
            ),
        )
        return None
    aggregates: list[ValidatedFidelityAggregateV3] = []
    for path in (original, swapped):
        aggregate = read_record(read(path)["comparisons"][0]["comparison"])
        if not isinstance(aggregate, ValidatedFidelityAggregateV3):
            raise ValueError("Order-swap artifact lacks a validated fidelity aggregate")
        aggregates.append(aggregate)
    if any(value.schedule.packet != packet for value in aggregates):
        raise ValueError("Order-swap audit changed the exact comparison packet")
    if aggregates[0].schedule.parser_versions != aggregates[1].schedule.parser_versions:
        raise ValueError("Order-swap audit changed the declared parser revisions")
    schedule = AnnotationScheduleV3(
        packet,
        tuple(slot for value in aggregates for slot in value.schedule.slots),
        aggregates[0].schedule.parser_versions,
        quorum=2,
        max_disagreement=0.0,
    )
    observations = tuple(item for value in aggregates for item in value.observations)
    combined = ValidatedFidelityAggregateV3(
        schedule, observations, aggregate_comparisons(observations, schedule=schedule)
    )
    labels = output / "labels.json"
    immutable(
        labels,
        dict(
            schema=FIDELITY_TRAINING_SCHEMA,
            aggregation_revision=AGGREGATION_REVISION,
            comparisons=[dict(packet=packet.to_dict(), comparison=combined.to_dict())],
        ),
    )
    write_artifact(
        output / "audit.json",
        dict(
            status="complete",
            comparison_id=comparison_id,
            scheduled=2,
            available=2,
            labels_sha256=sha(labels),
            global_target_eligible=combined.global_target_eligible,
        ),
    )
    return labels


def annotate_decoded(request_path: Path, manifest_path: Path, *, seconds: float) -> Path | None:
    """Attach fresh DEV labels only to the exact decoded, verified generated pool."""
    from exact.repair.workers import bounded_call
    from tools.repair.prepare import case_from_dict

    started = time.monotonic()
    request, manifest = read(request_path), validate_manifest(read(manifest_path))
    seconds = _remaining_seconds(seconds, manifest)
    if request["schema"] != "exact-repair/post-decode-annotation-request/v1" or (
        request["manifest_hash"] != sha(manifest_path)
    ):
        raise ValueError("Post-decode request/manifest binding changed")
    case = case_from_dict(request["case"])
    from exact.repair.kernel import _valid_report

    verification = read_record(request["verification"])
    if (
        not isinstance(verification, VerificationReportV2)
        or not verification.authorizes
        or not _valid_report(case.problem, tuple(request["assignment"]), verification)
    ):
        raise ValueError("Post-decode request lacks exact qualified verification")
    if case.split != "development" or manifest["phase"] != "development":
        raise ValueError("Post-decode training hook is restricted to DEV")
    selection_slot = request.get("selection_slot")
    slot = canonical_hash(selection_slot)
    frozen_slot = manifest.get("frozen_annotation_slots", {}).get(slot)
    if frozen_slot is None:
        return None
    if (
        frozen_slot["selection_slot"] != selection_slot
        or selection_slot.get("case_id") != case.case_id
    ):
        raise ValueError("Decoded output changed its preselected semantic slot")
    entry = manifest["cases"].get(case.case_id)
    if entry is None or seconds < 5:
        return None  # Missing frozen slot remains in the full DEV denominator.
    if (entry["parent"], entry["query_basis_hash"]) != (
        case.structural_parent,
        canonical_hash(consequence_basis_from_probes(case.probes)),
    ):
        raise ValueError("Decoded case changed its semantic evidence basis")
    # Canonical IDs survive generated-inventory reordering. No observed outcome
    # chooses a new counterpart or silently replaces a failed comparison slot.
    by_id = [
        {c.candidate_id: i for i, c in enumerate(obj.candidates)} for obj in case.problem.objects
    ]
    counterpart = entry["counterpart_candidate_ids"]
    if len(counterpart) != len(by_id) or any(
        c not in lookup for c, lookup in zip(counterpart, by_id)
    ):
        return None
    assignments = (
        tuple(request["assignment"]),
        tuple(lookup[c] for c, lookup in zip(counterpart, by_id)),
    )
    if assignments[0] == assignments[1]:
        return None
    plans: list[SemanticPlanV3] = []
    from exact.repair.kernel import materialize

    for assignment in assignments:
        context_hash = canonical_hash(
            (
                case.case_id,
                case.structural_parent,
                case.split,
                materialize(case.problem, assignment),
                case.problem.policy.content_hash,
                consequence_basis_from_probes(case.probes),
                tuple(
                    (obj.object_id, obj.candidates[i].candidate_id)
                    for obj, i in zip(case.problem.objects, assignment)
                ),
            )
        )
        saved_plan = request_path.parent / "verified-plans" / (canonical_hash(assignment) + ".json")
        if saved_plan.exists():
            committed = read(saved_plan)
            if committed["context_hash"] != context_hash:
                raise ValueError("Committed annotation plan context changed")
            plan = read_record(committed["plan"])
            if not isinstance(plan, SemanticPlanV3):
                raise ValueError("Committed annotation artifact is not a verified semantic plan")
            plans.append(plan)
            continue
        left = seconds - (time.monotonic() - started)
        if left <= 4:
            return None
        outcome = bounded_call(
            _verified_plan,
            request["case"],
            assignment,
            timeout=min(30, left - 2),
            memory_mb=manifest.get("verification_memory_mb", 8192),
        )
        if outcome.status != "complete" or not outcome.cleanup_complete or outcome.value is None:
            return None
        if not isinstance(outcome.value, SemanticPlanV3):
            raise ValueError("Annotation verifier returned an invalid semantic plan record")
        immutable(saved_plan, dict(context_hash=context_hash, plan=outcome.value.to_dict()))
        plans.append(outcome.value)
    packet = SemanticEvidencePacketV3(
        case.case_id,
        case.structural_parent,
        case.split,
        entry["task"],
        tuple(entry["original_observation"]),
        entry["evidence"],
        tuple(entry["local_context"]),
        plans[0],
        plans[1],
        manifest["rubric_version"],
        manifest["criterion_weights"],
        tuple(p.probe_id for p in case.probes),
        entry["coverage"],
    )
    return annotate_comparison(
        packet,
        manifest,
        request_path.parent / "annotation",
        comparison_id=slot,
        order_swap_audit=bool(frozen_slot.get("swapped", False)),
        seconds=seconds - (time.monotonic() - started),
    )


def run(manifest_path, output):
    manifest = validate_manifest(read(manifest_path))
    deferrals = _provider_deferrals(manifest)
    exclusions = _calibration_exclusions(manifest)
    if manifest["phase"] == "calibration":
        path = Path(manifest_path).resolve()
        immutable(
            path.parent / "used.json",
            dict(
                schema="exact-repair/calibration-consumption/v1",
                manifest_path=str(path),
                manifest_sha256=sha(path),
            ),
        )
    admitted = None
    if manifest.get("packet_admission"):
        from tools.repair.train_packet_admission import validate_rows

        admitted = validate_rows(manifest)
    elif any(
        row.get("status") or ("packet" in row and row["packet"] is None)
        for row in manifest["slots"]
    ):
        raise ValueError("Closed annotation rows require a bound packet admission contract")
    started = time.monotonic()
    rows = []

    def unavailable(row):
        return dict(
            id=row["id"],
            status=row["status"],
            artifact=None,
            original_status=row["original_status"],
            original_row=row["original_row"],
            attempted=False,
        )

    def pending(row):
        if admitted is not None and row["status"] != "eligible":
            return unavailable(row)
        return dict(id=row["id"], status="not_attempted_provider_failure", artifact=None)

    def save(failure=None):
        write_artifact(
            Path(output) / "report.json",
            dict(
                schema="exact-repair/corrective-annotation-report/v1",
                scheduled=len(manifest["slots"]),
                recorded=len(rows),
                rows=rows,
                provider_policy_deferrals=deferrals,
                calibration_exclusions=exclusions,
                **(
                    _failure_report_fields(failure)
                    if failure
                    else dict(
                        status="complete" if len(rows) == len(manifest["slots"]) else "running"
                    )
                ),
            ),
        )

    eligible = admitted is None or any(r["status"] == "eligible" for r in manifest["slots"])
    failure = _authentication_preflight(manifest, output) if eligible else None
    if failure:
        rows = [pending(row) for row in manifest["slots"]]
        write_artifact(
            Path(output) / "report.json",
            dict(
                schema="exact-repair/corrective-annotation-report/v1",
                scheduled=len(rows),
                recorded=len(rows),
                rows=rows,
                **_failure_report_fields(failure),
            ),
        )
        raise ConfirmedAnnotationFailure(failure)
    for index, row in enumerate(manifest["slots"]):
        if admitted is not None and row["status"] != "eligible":
            rows.append(unavailable(row))
            save()
            continue
        profile = row.get("profile", manifest["profile"])
        deferred = deferrals.get(profile)
        excluded = exclusions.get(profile)
        packet_path = row["packet"]
        if sha(packet_path["path"]) != packet_path["sha256"]:
            raise ValueError("Frozen annotation packet changed")
        packet = read_record(read(packet_path["path"]))
        left = _remaining_seconds(manifest["seconds"] - (time.monotonic() - started), manifest)
        failure = None
        try:
            artifact = (
                annotate_packet(
                    packet,
                    {**manifest, "profile": profile},
                    Path(output) / row["id"],
                    slot_id=row["id"],
                    swapped=row.get("swapped", False),
                    seconds=left,
                    comparison_id=row.get("comparison_id", row["id"]),
                )
                if left > 3 and deferred is None and excluded is None
                else None
            )
        except ConfirmedAnnotationFailure as error:
            failure = error
            artifact = None
            if manifest["phase"] == "calibration" and _provider_policy_failure(
                manifest, error.evidence
            ):
                deferred = deferrals[profile] = dict(
                    failed_slot=row["id"], confirmed_failure=error.evidence
                )
                failure = None
        policy_failure = deferred and deferred["failed_slot"] == row["id"]
        rows.append(
            (
                excluded["retained_rows"].get(
                    row["id"],
                    dict(id=row["id"], status="not_attempted_profile_ineligible", artifact=None),
                )
            )
            if excluded
            else dict(
                id=row["id"],
                status=(
                    "provider_error"
                    if failure or policy_failure
                    else (
                        "not_attempted_provider_policy"
                        if deferred
                        else "complete" if artifact else "unavailable"
                    )
                ),
                artifact=str(artifact) if artifact else None,
                **(
                    dict(confirmed_failure=failure.evidence)
                    if failure
                    else (
                        dict(confirmed_failure=deferred["confirmed_failure"])
                        if policy_failure
                        else {}
                    )
                ),
            )
        )
        if failure:
            rows.extend(pending(tail) for tail in manifest["slots"][index + 1 :])
        save(failure.evidence if failure else None)
        if failure:
            raise failure
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
