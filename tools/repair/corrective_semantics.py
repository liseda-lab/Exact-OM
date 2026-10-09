"""Small, bounded semantic annotation runner shared by acquisition and decoded DEV.

Evidence and pair slots are frozen before annotation. The existing adapter owns
wire provenance; this module adds campaign/phase quotas and post-decode binding.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, canonical_json, read_record
from exact.repair.semantic_fidelity import (
    AGGREGATION_REVISION,
    CRITERIA,
    EVALUATOR,
    FIDELITY_TRAINING_SCHEMA,
    TEACHER,
    AnnotationBudget,
    AnnotationScheduleV3,
    SelectionAnnotationRunV1,
    SemanticAnnotationAdapter,
    SemanticConsequenceReportV3,
    SemanticEvidencePacketV3,
    ValidatedFidelityAggregateV3,
    aggregate_comparisons,
    consequence_basis_from_probes,
    semantic_plan_from_verification,
)
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import immutable


PHASE_LIMITS = {"calibration": 32, "train": 384, "development": 96, "test": 192}
COMPARISON_LIMITS = {"calibration": 32, "train": 256, "development": 64, "test": 128}


class AnnotationBudgetExhausted(ValueError):
    """Expected terminal masking at a frozen cumulative quota, never a retry."""


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
    if not 0 < value["cost_ceiling_usd"] <= 35:
        raise ValueError("The authorized campaign ceiling is $35")
    if value["phase"] not in PHASE_LIMITS:
        raise ValueError("Unregistered annotation phase")
    if value["request_limits"] != PHASE_LIMITS:
        raise ValueError("Request quotas require an explicit protocol amendment")
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
    return value


def _reserve_phase(manifest, slot, packet_hash, cost, *, comparison_id=None):
    """Fail closed after a killed sender; replacement jobs never replenish quotas."""
    path = Path(manifest["ledger_directory"]) / "phase-reservations.json"
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
        if (state["lineage"], state["cost_ceiling_usd"], state["request_limits"]) != (
            manifest["lineage_id"],
            manifest["cost_ceiling_usd"],
            PHASE_LIMITS,
        ):
            raise ValueError("Annotation budget lineage changed")
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
        unique = {row.get("comparison_id", row["slot"]) for row in same}
        if comparison_id not in unique and len(unique) >= COMPARISON_LIMITS[manifest["phase"]]:
            raise AnnotationBudgetExhausted("Cumulative unique comparison quota exhausted")
        if len(same) >= PHASE_LIMITS[manifest["phase"]] or len(records) >= 704:
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
        return artifact if receipt["status"] == "complete" else None
    # The request configuration is frozen across restart. A shorter remaining
    # slice is an unavailable slot, never a changed identity or hidden retry.
    if _remaining_seconds(seconds, manifest) < 92 or not packet.eligible:
        return None
    profile_name = manifest["profile"]
    profile = manifest["profiles"][profile_name]
    prices = manifest["prices_per_million"][profile_name]
    # Conservative maximum includes billed reasoning tokens in the output cap.
    cost = (8000 * prices["input"] + 2000 * prices["output"]) / 1_000_000
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
    run = SelectionAnnotationRunV1(
        run_id=canonical_hash((identity, "run")),
        lineage_id=manifest["lineage_id"],
        authorized=True,
        role_profiles={TEACHER: selected_teacher, EVALUATOR: evaluator},
        role_budgets=budgets,
        aggregate_budget=AnnotationBudget(704, 704 * 10000, manifest["cost_ceiling_usd"], 704 * 90),
        max_input_bytes=8000,
        max_output_tokens=2000,
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
    if len(json.dumps(json.loads(canonical_json(messages))).encode()) + 256 > 8000:
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
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="unknown_delivery",
                costs_reset=False,
                retry_permitted=False,
            ),
        )
        router.hosted.close()
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
        write_artifact(
            receipt_path,
            dict(
                identity=identity,
                status="annotation_unavailable",
                error=str(error),
                costs=adapter.summary(),
                retry_permitted=False,
                costs_reset=False,
            ),
        )
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


def annotate_comparison(packet, manifest, output, *, comparison_id, order_swap_audit, seconds):
    """Original plus a scheduled audit; missing or discordant audits cannot label."""
    started = time.monotonic()
    output = Path(output)
    original = annotate_packet(
        packet,
        manifest,
        output / "original",
        slot_id=comparison_id + ":original",
        comparison_id=comparison_id,
        seconds=seconds,
    )
    if original is None or not order_swap_audit:
        return original
    swapped = annotate_packet(
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
    aggregates = [
        read_record(read(path)["comparisons"][0]["comparison"]) for path in (original, swapped)
    ]
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
    if not verification.authorizes or not _valid_report(
        case.problem, tuple(request["assignment"]), verification
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
    plans = []
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
            plans.append(read_record(committed["plan"]))
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
        *plans,
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
    started = time.monotonic()
    rows = []
    for row in manifest["slots"]:
        packet_path = row["packet"]
        if sha(packet_path["path"]) != packet_path["sha256"]:
            raise ValueError("Frozen annotation packet changed")
        packet = read_record(read(packet_path["path"]))
        left = _remaining_seconds(manifest["seconds"] - (time.monotonic() - started), manifest)
        artifact = (
            annotate_packet(
                packet,
                {**manifest, "profile": row.get("profile", manifest["profile"])},
                Path(output) / row["id"],
                slot_id=row["id"],
                swapped=row.get("swapped", False),
                seconds=left,
                comparison_id=row.get("comparison_id", row["id"]),
            )
            if left > 3
            else None
        )
        rows.append(
            dict(
                id=row["id"],
                status="complete" if artifact else "unavailable",
                artifact=str(artifact) if artifact else None,
            )
        )
        write_artifact(
            Path(output) / "report.json",
            dict(
                schema="exact-repair/corrective-annotation-report/v1",
                scheduled=len(manifest["slots"]),
                recorded=len(rows),
                rows=rows,
                status="complete" if len(rows) == len(manifest["slots"]) else "running",
            ),
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
