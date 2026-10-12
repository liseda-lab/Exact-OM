"""Versioned categorical DEV observations over authenticated, saved native plans.

The pilot's exact prompt/anchor mapping is reused. This adapter has a separate
role and request identity, immutable wire receipts, no retries, and never imports
old numeric judgments. Selection uses the original frozen training projection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from collections import Counter
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from exact.llm.ledger import RequestLedger
from exact.llm.routing import LLMProfile, OpenRouterClient, extract_chat_text
from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from exact.repair.semantic_fidelity import SemanticEvidencePacketV3, aggregate_plan_ratings
from tools.repair import judge_pilot as pilot
from tools.repair.batch import locked, read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable

VERSION = "exact-repair/categorical-dev/v1"
ROLE = "repair-categorical-development-selection/v1"


def packet_for(slot):
    packet = read_record(bound(slot["packet"]))
    if (
        not isinstance(packet, SemanticEvidencePacketV3)
        or packet.split != "development"
        or not packet.eligible
    ):
        raise ValueError("Categorical DEV requires complete verified development evidence")
    if set(packet.criterion_weights) != set(pilot.CRITERIA) or any(
        not math.isclose(v, 1 / 3) for v in packet.criterion_weights.values()
    ):
        raise ValueError("Pilot prompt requires the unchanged three equal criterion weights")
    return packet


def payload(plan, slot):
    packet = packet_for(slot)
    model = plan["model"]
    return dict(
        model=model["profile"]["model"],
        messages=[
            dict(role="system", content=pilot.PROMPT),
            dict(
                role="user",
                content=json.dumps(
                    packet.judge_payload(swapped=slot["swapped"]), ensure_ascii=False
                ),
            ),
        ],
        provider=model["profile"]["provider"],
        max_tokens=6000,
        reasoning=model["reasoning"],
        response_format=dict(
            type="json_schema",
            json_schema=dict(
                name="repair_categorical_dev",
                strict=True,
                schema=pilot.response_schema(packet.evidence),
            ),
        ),
    )


def assess(answer, packet, swapped=False):
    # Reuse the strict pilot structural validator, without construction gold or
    # pretending that production preferences have known correct answers.
    neutral = dict(
        preference=answer.get("preference") if isinstance(answer, dict) else None,
        allowed_anchors={
            k: dict(a=list(pilot.ANCHORS), b=list(pilot.ANCHORS)) for k in pilot.CRITERIA
        },
    )
    check = pilot.assess(answer, pilot.response_schema(packet.evidence), neutral)
    errors = []
    if not check["cited"]:
        errors.append("missing_rationale_or_citation")
    if not check["preference_consistent"]:
        errors.append("preference_numeric_conflict")
    plans = (packet.plan_b, packet.plan_a) if swapped else (packet.plan_a, packet.plan_b)
    if plans[0].content[0] == plans[1].content[0] and any(
        row["a"] != row["b"] for row in answer["criteria"].values()
    ):
        errors.append("identical_theory_anchor_conflict")
    for side, selected in zip(("a", "b"), plans):
        facts = [
            f
            for f in packet.evidence.values()
            if f["kind"] == "native_intended_entailed"
            and json.loads(f["text"])["plan_id"] == selected.plan_id
        ]
        if answer["criteria"]["meaning_retention"][side] == "full" and any(
            f["symbolic_value"] == "false" for f in facts
        ):
            errors.append("perfect_retention_with_missing_commitment:" + side)
    eligible = not errors and answer["preference"] != "abstain"
    return dict(
        eligible=eligible,
        errors=errors,
        decision=answer["preference"],
        scores=check["scores"],
        criteria=answer["criteria"],
        masks=dict(
            anchored_value=eligible,
            ranking=eligible and answer["preference"] in ("A", "B"),
            tie=eligible and answer["preference"] == "tie",
        ),
        unknown=any(r[s] == "unknown" for r in answer["criteria"].values() for s in ("a", "b")),
    )


def request_cost(plan, slot):
    wire = payload(plan, slot)
    cost = pilot.request_cost(plan["model"], wire, plan["max_input_bytes"])
    endpoint = bound(plan["model"]["endpoint"])
    route = next(r for r in endpoint["data"]["endpoints"] if r["tag"] == plan["model"]["route"])
    prices = [route["pricing"]] + [
        p
        for p in route["pricing"].get("overrides", [])
        if p["min_prompt_tokens"] <= cost["input_token_upper_bound"]
    ]
    # Bound cache creation as an additional charge, without optimistic discounts.
    cache = max(float(p.get("input_cache_write", 0)) for p in prices)
    cost["cache_write_usd_per_token"] = cache
    cost["reserved_cost_usd"] += cache * cost["input_token_upper_bound"]
    return cost


def validate(plan):
    authorization = bound(plan["authorization"])
    selection = bound(authorization["teacher_selection"])
    prior = bound(authorization["prior_development_plan"])
    manifest = bound(prior["annotation_manifest"])
    amendment = bound(manifest["dev_interface_amendment"])
    quota = bound(plan["quota_amendment"])
    if (
        plan["schema"] != VERSION
        or plan["prompt"] != pilot.PROMPT
        or plan["model"] != selection["profile"]
        or selection["selected_teacher"] != "luna"
        or selection["selection_used_gemini"] is not False
        or plan["models"] != prior["models"]
        or plan["semantic_role"] != ROLE
        or plan["deadline_epoch"] > 1791997200
        or plan["budget_phase"] != "development"
        or quota["new_limits"] != authorization["budget"]["new_limits"]
        or sum(quota["new_limits"].values()) != 704
        or quota["cost_ceiling_usd"] != 35
        or plan["timeout_seconds"] != 180
        or plan["max_output_tokens"] != 6000
    ):
        raise ValueError("Categorical successor changed its authorized protocol")
    expected = {
        (key, swap)
        for key in amendment["requests"]
        for swap in (
            [False, True] if manifest["frozen_annotation_slots"][key].get("swapped") else [False]
        )
    }
    seen, ids, costs = set(), set(), {}
    for slot in plan["slots"]:
        key = slot["selection_key"]
        identity = (key, slot["swapped"])
        if identity in seen or slot["id"] in ids or identity not in expected:
            raise ValueError("Duplicate or foreign categorical presentation")
        seen.add(identity)
        ids.add(slot["id"])
        request = bound(slot["source_request"])
        member = amendment["requests"][key]
        if (
            slot["source_request"] != member["source_request"]
            or canonical_hash(request["selection_slot"]) != key
            or slot["selection_slot"] != request["selection_slot"]
        ):
            raise ValueError("Categorical observation changed decoded plan provenance")
        packet = packet_for(slot)
        if packet.case_id != request["case"]["case"]["case_id"]:
            raise ValueError("Packet case mismatch")
        old_request = bound(slot["packet_request"])
        if old_request["interface_origin"] != member["source_request"]:
            raise ValueError("Saved native packet has a foreign request origin")
        # Reuse exact packet records produced by the authenticated full preflight.
        if Path(slot["packet"]["path"]).parent != Path(slot["packet_request"]["path"]).parent:
            raise ValueError("Packet and native request lineage differ")
        for ref in member["verified_plans"]:
            bound(ref)
        costs[slot["id"]] = request_cost(plan, slot)
    if seen != expected or len(seen) != plan["max_requests"]:
        raise ValueError("Categorical successor lost a fixed presentation")
    native_admission = bound(plan["source_native_admission"])
    if (
        native_admission["manifest"] != prior["annotation_manifest"]
        or native_admission["truncated"]
        or {(r["slot"], r["swapped"]) for r in native_admission["rows"]} != expected
        or any(r["status"] != "eligible" for r in native_admission["rows"])
    ):
        raise ValueError("Prior complete native preflight changed")
    test = bound(quota["test_schedule"])
    test_rows = test["semantic_slots"]
    conference = Counter(r["seed"] for r in test_rows if r["cohort"] == "conference")
    if (
        len(test_rows) != 128
        or len({r["id"] for r in test_rows}) != 128
        or sum(r["swapped_audit"] for r in test_rows) != 26
        or sum(conference.values()) != 43
        or max(conference.values()) - min(conference.values()) > 1
        or quota["new_limits"]["test"] < 154
        or test["independent_judge"] != "google/gemini-2.5-pro"
        or test["test_outcomes_opened"]
    ):
        raise ValueError("Quota reallocation changed the frozen TEST denominator")
    for model in plan["models"]:
        weights = bound(model["weights"])
        authenticate(weights["artifact"])
        report = bound(model["source_report"])
        if (
            weights["epoch"] != 6
            or weights["optimizer_updates"] != 756
            or len(report["rows"]) != 32
        ):
            raise ValueError("Saved six-model cohort changed")
        for ref in report["rows"]:
            bound(ref)
        for slot in plan["slots"]:
            if (
                slot["selection_slot"]["seed"] == model["seed"]
                and slot["selection_slot"]["supervision_condition"] == model["condition"]
            ):
                validate_decoded_identity(slot, weights)
    return costs


def validate_decoded_identity(slot, recovery):
    from exact.repair.kernel import materialize
    from exact.repair.semantic_fidelity import consequence_basis_from_probes
    from tools.repair.prepare import case_from_dict

    request = bound(slot["source_request"])
    packet = packet_for(slot)
    case = case_from_dict(request["case"])
    selected = {
        o.object_id: o.candidates[i].candidate_id
        for o, i in zip(case.problem.objects, request["assignment"])
    }
    if (
        request["model_hash"] != recovery["artifact"]["model_hash"]
        or packet.parent_group_id != case.structural_parent
        or case.split != "development"
        or packet.plan_a.policy_hash != case.problem.policy.content_hash
        or packet.plan_a.query_basis_hash
        != canonical_hash(consequence_basis_from_probes(case.probes))
        or packet.plan_a.theory_hash
        != canonical_hash(materialize(case.problem, tuple(request["assignment"])))
        or dict(packet.plan_a.complete_assignment) != selected
    ):
        raise ValueError("Exact decoded-plan label identity mismatch at preflight")


def reserve(plan, costs):
    previous = bound(plan["previous_phase_state"])
    quota = bound(plan["quota_amendment"])
    path = Path(plan["ledger_directory"]) / "phase-reservations.json"
    with locked(path.with_suffix(".lock")):
        state = read(path)
        if (
            state["request_limits"] != quota["new_limits"]
            or state["cost_ceiling_usd"] != 35
            or state["lineage"] != previous["lineage"]
            or any(state["reservations"].get(k) != v for k, v in previous["reservations"].items())
        ):
            raise ValueError("Historical reservations or amended quota changed")
        expected = {
            canonical_hash(("development", s["id"])): dict(
                phase="development",
                slot=s["id"],
                packet_hash=s["packet"]["sha256"],
                request_basis=canonical_hash(plan),
                comparison_id=s["selection_key"],
                reserved_cost_usd=costs[s["id"]]["reserved_cost_usd"],
                purpose=ROLE,
            )
            for s in plan["slots"]
        }
        fresh = {k: v for k, v in expected.items() if k not in state["reservations"]}
        for key, row in expected.items():
            if key in state["reservations"] and any(
                state["reservations"][key].get(k) != v for k, v in row.items()
            ):
                raise ValueError("Categorical resume changed reservation identity")
        rows = list(state["reservations"].values()) + list(fresh.values())
        if (
            len(rows) > 704
            or sum(r["reserved_cost_usd"] for r in rows) > 35
            or any(
                sum(r["phase"] == phase for r in rows) > limit
                for phase, limit in quota["new_limits"].items()
            )
        ):
            raise ValueError("Complete categorical cohort exceeds cumulative budget")
        for key, row in fresh.items():
            state["reservations"][key] = dict(row, state="reserved", admitted_epoch=time.time())
        write_artifact(path, state)


def send(profile, wire, directory, role):
    """Run inside the bounded sender; the shared SQLite ledger commits raw bytes."""
    client = OpenRouterClient()
    client.ledger_dir = Path(directory)
    client.max_retries = 0
    client.retry_unknown_requests = False
    try:
        return client._generation(profile, wire, "chat/completions", role=role, requested_seed=None)
    finally:
        client.close()


def wire_identity(profile, wire, identity):
    return dict(
        schema_version=1,
        role=ROLE + ":" + identity,
        endpoint=profile.api_base + "/chat/completions",
        payload=wire,
        revision=profile.revision,
        tokenizer=profile.tokenizer,
        tokenizer_revision=profile.tokenizer_revision,
        requested_seed=None,
    )


def verify_receipt(plan, slot, record):
    from exact.llm.ledger import request_identity

    identity = canonical_hash((plan, slot))
    profile = LLMProfile.from_raw("luna", plan["model"]["profile"])
    key = request_identity(wire_identity(profile, payload(plan, slot), identity))[0]
    if (
        record["identity"] != identity
        or record["request_id"] != key
        or record["attempt"] != 1
        or record["role"] != ROLE
        or record["packet"] != slot["packet"]
    ):
        raise ValueError("Categorical receipt provenance changed")
    if record.get("response"):
        response = bound(record["response"])
        raw = RequestLedger(Path(plan["ledger_directory"])).cached(key)
        if (
            raw is None
            or json.loads(raw) != response
            or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]
        ):
            raise ValueError("Categorical response differs from durable HTTP receipt")
        if "assessment" in record:
            if (
                response.get("model") not in plan["model"]["accepted_model_ids"]
                or response.get("provider") != plan["model"]["provider_name"]
                or response["choices"][0]["finish_reason"] != "stop"
                or record["assessment"]
                != assess(
                    json.loads(extract_chat_text(response)), packet_for(slot), slot["swapped"]
                )
            ):
                raise ValueError("Saved categorical assessment changed")
    elif "assessment" in record:
        raise ValueError("Assessment has no durable response")
    return record


def one(plan, slot, output):
    directory = Path(output) / slot["id"]
    with locked(directory.with_suffix(".lock")):
        identity = canonical_hash((plan, slot))
        wire = payload(plan, slot)
        receipt = directory / "receipt.json"
        if receipt.exists():
            saved = read(receipt)
            return verify_receipt(plan, slot, saved)
        profile = LLMProfile.from_raw("luna", plan["model"]["profile"])
        ledger = RequestLedger(Path(plan["ledger_directory"]))
        request_id = ledger.plan(wire_identity(profile, wire, identity))
        immutable(
            directory / "request.json", dict(identity=identity, payload=wire, request_id=request_id)
        )
        marker = directory / "transmission-claimed.json"
        record = dict(
            identity=identity,
            slot=slot["id"],
            selection_key=slot["selection_key"],
            packet=slot["packet"],
            request_id=request_id,
            attempt=1,
            swapped=slot["swapped"],
            role=ROLE,
            status="unavailable",
            retry_permitted=False,
        )
        start = time.monotonic()
        try:
            if marker.exists():
                if read(marker) != dict(identity=identity, request_id=request_id):
                    raise ValueError("Transmission claim changed")
                if ledger.cached(request_id) is None:
                    raise RuntimeError("Unknown prior delivery; retransmission prohibited")
            deadline = min(
                plan["deadline_epoch"],
                float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", plan["deadline_epoch"])),
            )
            if time.time() + 190 > deadline and ledger.cached(request_id) is None:
                raise TimeoutError("A full request does not fit the remaining worker deadline")
            immutable(marker, dict(identity=identity, request_id=request_id))
            from exact.repair.workers import bounded_call

            outcome = bounded_call(
                send,
                profile,
                wire,
                plan["ledger_directory"],
                ROLE + ":" + identity,
                timeout=min(180, deadline - time.time() - 10),
                cleanup_seconds=2,
            )
            record["sender_cleanup_complete"] = outcome.cleanup_complete
            record["sender_status"] = outcome.status
            if outcome.status != "complete" or not outcome.cleanup_complete:
                if outcome.cleanup_complete:
                    with ledger._transaction() as db:
                        pending = db.execute(
                            "SELECT number FROM attempts WHERE request_id=? AND state='sent'",
                            (request_id,),
                        ).fetchall()
                    for attempt in pending:
                        ledger.unknown(
                            request_id,
                            attempt["number"],
                            "Bounded categorical sender ended without response",
                        )
                raise RuntimeError("Bounded sender: " + outcome.status)
            response = outcome.value
            immutable(directory / "response.json", response)
            raw = ledger.cached(request_id)
            if raw is None or json.loads(raw) != response:
                raise ValueError("Missing matching durable wire receipt")
            record.update(
                response=binding(directory / "response.json"),
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                usage=response.get("usage", {}),
                status="received",
            )
            if (
                response.get("model") not in plan["model"]["accepted_model_ids"]
                or response.get("provider") != plan["model"]["provider_name"]
                or response["choices"][0]["finish_reason"] != "stop"
            ):
                raise ValueError("Response route mismatch or incomplete output")
            answer = json.loads(extract_chat_text(response))
            record.update(
                answer=answer, assessment=assess(answer, packet_for(slot), slot["swapped"])
            )
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)[:300]
        record["elapsed_seconds"] = time.monotonic() - start
        immutable(receipt, record)
        phase = Path(plan["ledger_directory"]) / "phase-reservations.json"
        with locked(phase.with_suffix(".lock")):
            state = read(phase)
            row = state["reservations"][canonical_hash(("development", slot["id"]))]
            if row["request_basis"] != canonical_hash(plan):
                raise ValueError("Reservation changed at settlement")
            row["state"] = "completed" if record["status"] == "received" else "unresolved"
            write_artifact(phase, state)
        return record


def aggregate(slots, records, packet, policy):
    by_id = {r["slot"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {s["id"] for s in slots}:
        raise ValueError("Aggregation requires each scheduled observation exactly once")
    ratings, votes, observations, scores_a, scores_b = [], [], [], [], []
    for slot in slots:
        row = by_id[slot["id"]]
        if row["selection_key"] != slot["selection_key"] or row["swapped"] != slot["swapped"]:
            raise ValueError("Order/selection provenance mismatch")
        assessment = row.get("assessment", {})
        if not assessment.get("eligible"):
            continue
        side = "b" if slot["swapped"] else "a"
        decision = assessment["decision"]
        if slot["swapped"]:
            decision = {"A": "B", "B": "A"}.get(decision, decision)
        votes.append(decision)
        a, b = assessment["scores"]
        scores_a.append(b if slot["swapped"] else a)
        scores_b.append(a if slot["swapped"] else b)
        provenance = dict(
            request_id=row["request_id"],
            attempt=row["attempt"],
            response_sha256=row["raw_sha256"],
        )
        observations.append(provenance)
        ratings.append(
            dict(
                context=packet.content_hash,
                provenance=provenance,
                criteria={k: pilot.ANCHORS[v[side]] for k, v in assessment["criteria"].items()},
            )
        )
    if len({(r["request_id"], r["attempt"]) for r in observations}) != len(observations):
        raise ValueError("Duplicate wire observation")
    # Unchanged schedule: quorum one for ordinary pairs, two for audited
    # pairs; zero dissent and scalar medians must agree with the unique vote.
    median_a = statistics.median(scores_a) if scores_a else None
    median_b = statistics.median(scores_b) if scores_b else None
    numeric = (
        None
        if median_a is None
        else ("tie" if abs(median_a - median_b) <= 1e-6 else "A" if median_a > median_b else "B")
    )
    eligible = len(votes) == len(slots) and len(set(votes)) == 1 and numeric == votes[0]
    rating_aggregate = (
        aggregate_plan_ratings(ratings, packet.criterion_weights, policy)
        if policy is not None
        else None
    )
    status = (
        "comparison_abstention"
        if not eligible
        else rating_aggregate["status"] if rating_aggregate else "eligible"
    )
    score = (
        None
        if status != "eligible"
        else rating_aggregate["score"] if rating_aggregate else median_a
    )
    # The core plan-rating record retains its own exact schema/hash. The outer
    # schedule/audit receipt has a distinct schema and binds the raw HTTP bytes.
    result = dict(
        schema="exact-repair/categorical-dev-aggregate/v1",
        packet_hash=packet.content_hash,
        status=status,
        score=score,
        plan_rating_aggregate=rating_aggregate,
        numeric_rule="median",
        overall_score_a=median_a,
        overall_score_b=median_b,
        scheduled=len(slots),
        quorum=len(slots),
        max_disagreement=0.0,
        observed=len(records),
        eligible_observations=len(votes),
        invalid_or_missing=len(slots) - len(votes),
        votes=votes,
        observations=observations,
    )
    return dict(result, receipt_hash=canonical_hash(result))


def select_models(plan, output, records):
    from exact.repair.protocol import load_protocol_v3, training_projection_v3
    from exact.repair.learning import generated_checkpoint_criterion
    from exact.repair.kernel import materialize
    from exact.repair.semantic_fidelity import consequence_basis_from_probes
    from tools.repair.prepare import case_from_dict
    from tools.repair.validation_recovery import runtime_configuration

    prior = bound(bound(plan["authorization"])["prior_development_plan"])
    amendment = bound(bound(prior["annotation_manifest"])["dev_interface_amendment"])
    historical = bound(amendment["historical_review"])
    if (
        not historical["training_integrity_passed"]
        or not historical["full_dev_lifecycle_qualified"]
    ):
        raise ValueError("Historical training/lifecycle not qualified")
    models = []
    for model in plan["models"]:
        recovery = bound(model["weights"])
        primary = bound(recovery["primary_manifest"])
        config = runtime_configuration(
            training_projection_v3(
                load_protocol_v3(authenticate(primary["protocol"]), for_execution=True)
            )
        )
        source = bound(model["source_report"])
        manifest = bound(source["manifest"])
        predecessor = next(
            m
            for m in bound(manifest["historical_review"])["models"]
            if m["seed"] == model["seed"] and m["condition"] == model["condition"]
        )
        if any(r.get("selection_criterion") is not None for r in predecessor["development"]):
            raise ValueError("Prior eligible checkpoint must participate in selection")
        own = [
            s
            for s in plan["slots"]
            if s["selection_slot"]["seed"] == model["seed"]
            and s["selection_slot"]["supervision_condition"] == model["condition"]
        ]
        semantic_ids = {s["selection_slot"]["case_id"] for s in own}
        results = []
        available = 0
        seen = set()
        directory = Path(output) / "models" / f"{model['condition']}-s{model['seed']}"
        for ref in source["rows"]:
            row = bound(ref)
            result = copy.deepcopy(row["result"])
            case_id = row["case_id"]
            seen.add(case_id)
            key = canonical_hash(row["selection_slot"])
            group = [s for s in own if s["selection_key"] == key]
            if group:
                request = bound(group[0]["source_request"])
                packet = packet_for(group[0])
                case = case_from_dict(request["case"])
                assignment = tuple(request["assignment"])
                selected = {
                    o.object_id: o.candidates[i].candidate_id
                    for o, i in zip(case.problem.objects, assignment)
                }
                if (
                    request["model_hash"] != recovery["artifact"]["model_hash"]
                    or request["selection_slot"] != row["selection_slot"]
                    or packet.parent_group_id != case.structural_parent
                    or case.split != "development"
                    or packet.plan_a.policy_hash != case.problem.policy.content_hash
                    or packet.plan_a.query_basis_hash
                    != canonical_hash(consequence_basis_from_probes(case.probes))
                    or packet.plan_a.theory_hash
                    != canonical_hash(materialize(case.problem, assignment))
                    or dict(packet.plan_a.complete_assignment) != selected
                ):
                    raise ValueError("Exact decoded-plan label identity mismatch")
                value = aggregate(
                    group,
                    [r for r in records if r["slot"] in {s["id"] for s in group}],
                    packet,
                    config["plan_rating_aggregation"],
                )
                immutable(directory / "labels" / (key + ".json"), value)
                score = value["score"]
                available += score is not None
                symbolic = request["symbolic_label"]
                mixture = config["mixture_symbolic_weight"]
                benefit = (
                    mixture * symbolic["benefit"] + (1 - mixture) * score
                    if score is not None and symbolic["benefit"] is not None
                    else None
                )
                result["decoded"].update(
                    status="verified" if benefit is not None else "unknown_selected_labels",
                    selected_utility=benefit - symbolic["cost"] if benefit is not None else None,
                    semantic_benefit=benefit,
                    weak_benefit=score,
                    symbolic_benefit=symbolic["benefit"],
                    annotation_pending=False,
                    annotation_artifact=str(directory / "labels" / (key + ".json")),
                )
            result["case_id"] = case_id
            immutable(directory / "rows" / (key + ".json"), dict(source=ref, result=result))
            results.append(result)
        if len(results) != 32 or seen != set(manifest["case_ids"]):
            raise ValueError("Lost full DEV case coverage")
        criterion = generated_checkpoint_criterion(
            results,
            expected_case_ids=manifest["case_ids"],
            semantic_case_ids=semantic_ids,
            minimum_coverage=config["minimum_generated_coverage"],
            uncertainty_z=config["quality_uncertainty_z"],
            fallback=config["missing_label_fallback"],
        )
        report = dict(
            schema=VERSION,
            source=model["source_report"],
            rows=32,
            epoch=6,
            selection_criterion=criterion,
            selected_checkpoint=recovery["artifact"] if criterion is not None else None,
            semantic_slots=len(semantic_ids),
            semantic_available=available,
            verified_dev_rows=sum(
                r["decoded"]["logical_status"] == "VERIFIED_FEASIBLE" for r in results
            ),
            selection_rule_unchanged=True,
            test_opened=False,
            costs_reset=False,
            plan_identity=canonical_hash(plan),
        )
        immutable(directory / "report.json", report)
        models.append(
            dict(
                seed=model["seed"],
                condition=model["condition"],
                report=binding(directory / "report.json"),
                **report,
            )
        )
    report = dict(
        schema=VERSION,
        status=(
            "qualified"
            if all(m["selected_checkpoint"] is not None for m in models)
            else "review_required"
        ),
        models=models,
        selected_models_available=sum(m["selected_checkpoint"] is not None for m in models),
        semantic_slots=sum(m["semantic_slots"] for m in models),
        semantic_available=sum(m["semantic_available"] for m in models),
        full_dev_rows=192,
        verified_dev_rows=sum(m["verified_dev_rows"] for m in models),
        training_integrity_passed=True,
        full_dev_lifecycle_qualified=all(m["verified_dev_rows"] == 32 for m in models),
        selection_rule_unchanged=True,
        test_opened=False,
        costs_reset=False,
        plan_identity=canonical_hash(plan),
        reported_cost_usd=sum(r.get("usage", {}).get("cost", 0) or 0 for r in records),
        manual_semantic_review_required=True,
    )
    immutable(Path(output) / "report.json", report)
    return report


def run(plan_path, output):
    plan = read(plan_path)
    output = Path(output)
    if (
        os.getenv("EXACT_EXPERIMENT_MODE") == "1"
        or os.getenv("EXACT_OPENROUTER_RETRY_UNKNOWN") == "1"
    ):
        raise ValueError("Incompatible inherited hosted controls")
    costs = validate(plan)
    immutable(output / "preflight.json", dict(plan=binding(plan_path), costs=costs))
    reserve(plan, costs)
    records = []
    # Preserve paired seed execution order; only independent presentations within
    # a matched seed run concurrently, at the existing four-request ceiling.
    for seed in (13, 37, 73):
        slots = [s for s in plan["slots"] if s["selection_slot"]["seed"] == seed]
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(lambda s: one(plan, s, output / "presentations"), slots))
        if any(r.get("sender_cleanup_complete") is False for r in records):
            raise RuntimeError("Categorical sender cleanup incomplete; preserve quarantine")
        write_artifact(
            output / "progress.json", dict(seed_completed=seed, presentations=len(records))
        )
    for slot in plan["slots"]:
        verify_receipt(plan, slot, next(r for r in records if r["slot"] == slot["id"]))
    return select_models(plan, output, records)


def review_receipts(plan_path, source, output):
    """Reaggregate saved authenticated wire responses, with no sending path."""
    plan = read(plan_path)
    validate(plan)
    records = []
    for slot in plan["slots"]:
        receipt = Path(source) / "presentations" / slot["id"] / "receipt.json"
        records.append(verify_receipt(plan, slot, read(receipt)))
    immutable(
        Path(output) / "source-receipts.json",
        dict(
            plan=binding(plan_path),
            hosted_calls=0,
            receipts=[
                binding(Path(source) / "presentations" / s["id"] / "receipt.json")
                for s in plan["slots"]
            ],
        ),
    )
    return select_models(plan, output, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--review-receipts", type=Path)
    args = parser.parse_args()
    if args.review_receipts:
        review_receipts(args.plan, args.review_receipts, args.output)
    elif args.preflight_only:
        costs = validate(read(args.plan))
        immutable(args.output, dict(plan=binding(args.plan), costs=costs))
    else:
        run(args.plan, args.output)


if __name__ == "__main__":
    main()
