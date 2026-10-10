"""Freeze controlled TRAIN evidence and check complete packets without hosted calls.

The authored setting is generator-dependent, not independent expert/domain truth.
Every frozen intention survives; native failures/oversize are terminal row outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import time

import pyowl_core as owl

from exact.repair.annotation_controls import input_token_bound, response_format
from exact.repair.records import canonical_hash, read_record
from exact.repair.owl import snapshot_from_axioms
from exact.repair.semantic_fidelity import SemanticEvidencePacketV3, annotation_prompt
from exact.repair.workers import bounded_call
from tools.repair.annotation_profile import input_limits, request_profile
from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read
from tools.repair.common_training import load_release
from tools.repair.corrective_campaign import source_identity
from tools.repair.corrective_semantics import _verified_plan
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict
from tools.repair.shared_release import (
    authenticate,
    bound,
    check_time,
    immutable,
    validate_completion,
)

REVISION = "authored-generator-semantics/20261010-v1"
# Explicit authored settings associated with the generator's eight constructors.
# These are stipulations for a fictional benchmark, not recovered real definitions.
SETTINGS = {
    "papers": "Two fictional submission catalogues distinguish papers, accepted papers and rejected papers. Acceptance evidence qualifies a submission for the accepted category; rejection is incompatible with that evidence. A general paper category is broader than accepted papers. Preserve the declared acceptance commitments without treating every paper as accepted.",
    "overlap": "A fictional source catalogue permits a person to hold both author and reviewer roles. The target catalogue's author and reviewer categories are exclusive administrative roles. Equal names across catalogues do not establish equal scope. Preserve the stated role commitments without erasing the source overlap category.",
    "disjointness": "A fictional source catalogue permits an overlapping author/reviewer role. The target's unrelated role must remain separate from each of those roles. The authored setting withdraws author-versus-reviewer exclusion but retains the independent exclusions involving the unrelated role.",
    "conjunct": "A fictional source has a category with an author commitment. The target author and reviewer roles are exclusive. The authored intended category retains its author commitment without requiring simultaneous reviewer membership.",
    "participant": "In a fictional event, participants include listeners and presenting participants. Listeners and presenting participants are separate; speakers are persons. The speaker commitment applies to presenting participants, not automatically to every participant. Preserve the declared listener and speaker correspondences.",
    "domain": "In a fictional review workflow, persons and software are distinct kinds of agent. Software can write a review. The writes relation therefore admits agents as subjects, without declaring every writer a person or equating software with persons.",
    "range": "In a fictional review workflow, persons and software are distinct kinds of agent. A review-related witness can be writtenBy software. The writtenBy relation admits agents as objects, without forcing software to be persons.",
    "filler": "In a fictional review workflow, persons and software are distinct kinds of agent. The writtenBy relation has a person range. The authored witness requires some agent filler, without insisting that this same filler be software. This preserves an existential commitment without forcing a software/person overlap.",
}
# This dictionary stipulates the fictional meanings of the constructor's explicit
# variable names. It is never applied to a publisher ontology or arbitrary IRI.
GLOSSARY = {
    "Paper_s": "a paper record in the source submission catalogue",
    "Rejected_s": "a source paper record carrying a rejection decision",
    "Accepted_t": "an accepted paper record in the target catalogue",
    "Rejected_t": "a rejected paper record in the target catalogue",
    "Paper_t": "a general paper record in the target catalogue",
    "Submission_s": "a submitted paper record with the declared acceptance evidence",
    "Acceptance_s": "membership evidence for a positive submission decision",
    "Author_s": "the source authorship role, which can overlap reviewing",
    "Reviewer_s": "the source reviewing role, which can overlap authorship",
    "Overlap_s": "the source category whose intended role commitments are declared below",
    "Author_t": "the target administrative author role",
    "Reviewer_t": "the target administrative reviewer role",
    "Unrelated_t": "an independent target role excluded from author and reviewer roles",
    "Participant_s": "a participant in the fictional event",
    "Speaker_s": "a person holding the event's speaker role",
    "Listener_s": "a participant holding the listener role",
    "Presenting_s": "a participant presenting at the event",
    "Person_s": "a human agent in the source workflow",
    "Listener_t": "the target counterpart of the listener role",
    "Speaker_t": "the target counterpart of the speaker role",
    "Witness_s": "a declared example category with the stated workflow commitments",
    "Software_s": "a software agent in the source workflow",
    "Agent_s": "a workflow agent, including persons and software",
    "Software_t": "the target counterpart of software agents",
    "Person_t": "the target counterpart of persons",
    "Review_s": "a review artifact in the source workflow",
    "writes": "the relation from a writer to a review artifact",
    "writtenBy": "the relation from a review-related item to its writer",
    "hasDecision0": "the relation carrying the declared decision evidence",
}


def glossary(case):
    entities = {e for a in case.intended_theory for e in owl.signature(a)}
    rows = []
    for entity in sorted(entities, key=owl.structural_hexdigest):
        iri = entity.iri.value
        if not iri.startswith("urn:exact:generated:"):
            raise ValueError("Controlled glossary cannot interpret external ontology IRIs")
        local = iri.rsplit(":", 1)[1]
        constructor_name = local
        if case.mirrored and local.endswith(("_s", "_t")):
            constructor_name = local[:-2] + ("_t" if local.endswith("_s") else "_s")
        if constructor_name in GLOSSARY:
            rows.append(iri + " := " + GLOSSARY[constructor_name])
        elif not __import__("re").fullmatch(
            r"(?:Core\d+_(?:branch\d+_step\d+|join)(?:_[st])?|Noise\d+)", constructor_name
        ):
            raise ValueError("Unknown constructor symbol has no authored meaning: " + local)
    return (
        "Explicit fictional symbol bindings (constructor roles follow its mirroring flag):\n"
        + "\n".join(rows)
    )


SCOPE = (
    "Explicit authored controlled setting, dependent on the original generator. "
    "Not an independently supplied definition, real Conference evidence, or expert truth. "
    "Names are local symbols: do not infer additional real-world facts from them. "
    "The complete fixed context appears in both reconstructed plans. Intermediate Core/Noise "
    "categories have only the formal membership constraints shown there. The intended editable "
    "commitments below specify the author's limited meaning target; they are not checked "
    "boolean query evidence and do not add or change symbolic probes."
)


def render(axioms):
    return owl.render_document(
        snapshot_from_axioms(tuple(axioms)).materialize().root, format="functional"
    ).decode()


def controlled_evidence(case, source_ref):
    if case.split != "train" or case.origin != "generated" or case.family not in SETTINGS:
        raise ValueError("Controlled evidence only supports the frozen generated TRAIN families")
    if not case.intended_theory:
        raise ValueError("Authored setting requires original intended construction")
    fixed = set(case.problem.fixed_axioms)
    if not fixed <= set(case.intended_theory):
        raise ValueError("Intended construction changed fixed context")
    # Bind exact evaluator bytes, not a reconstructed preferred assignment in the new pool.
    record = case_to_dict(case)
    remaining = [a for a in case.intended_theory if a not in fixed]
    evidence = {
        "D1": dict(
            source_id=source_ref["sha256"] + ":" + case.family,
            release=REVISION,
            kind="authored_controlled_setting",
            text=SCOPE + "\n" + SETTINGS[case.family],
            symbolic_value=None,
        ),
        "D2": dict(
            source_id=canonical_hash([owl.canonical_bytes(a).hex() for a in case.intended_theory])
            + ":intended_theory_minus_fixed",
            release=REVISION,
            kind="authored_construction_commitments",
            text="Intended editable commitments, together with the unchanged full fixed context:\n"
            + render(remaining),
            symbolic_value=None,
        ),
    }
    evidence["D3"] = dict(
        source_id=source_ref["sha256"] + ":constructor-symbol-bindings",
        release=REVISION,
        kind="authored_definition",
        text=glossary(case),
        symbolic_value=None,
    )
    return dict(
        schema="exact-repair/controlled-train-evidence/v1",
        revision=REVISION,
        case_id=case.case_id,
        parent=case.structural_parent,
        split=case.split,
        family=case.family,
        control=case.control,
        input_hash=case.problem.content_hash,
        evaluator_hash=record["hash"],
        source=source_ref,
        evidence=evidence,
        independent_semantic_evidence=False,
        real_data_coverage=False,
        probe_targets_changed=False,
        candidate_inventory_changed=False,
    )


def packet_for(case, evidence, plans, manifest):
    return SemanticEvidencePacketV3(
        case.case_id,
        case.structural_parent,
        "train",
        "Assess retained meaning only under this explicitly authored fictional setting. "
        "Do not infer external domain facts or reward a named repair action. Use abstain when "
        "the declared setting cannot distinguish the complete plans; logical validity is separate.",
        (render(a for obj in case.problem.objects for a in obj.original_axioms),),
        evidence["evidence"],
        (),
        plans[0],
        plans[1],
        manifest["rubric_version"],
        manifest["criterion_weights"],
        tuple(p.probe_id for p in case.probes),
        dict(
            complete=True,
            omissions=[],
            stop_reason="complete_controlled_setting",
            byte_budget=32768,
        ),
    )


def packet_admission(packet, manifest, swapped):
    """Same wire envelope with a fixed-length run identity; no router/client creation."""
    controls = request_profile(manifest, manifest["profile"])
    token_cap, byte_cap = input_limits(controls)
    prompt = annotation_prompt(manifest["prompt_version"])
    context = dict(
        packet=packet.judge_payload(swapped=swapped),
        run_id="0" * 64,
        lineage_id=manifest["lineage_id"],
        prompt_version=manifest["prompt_version"],
        prompt_hash=canonical_hash(prompt),
        repetition=0,
        swapped=swapped,
        correction=0,
        correction_errors=[],
    )
    messages = [
        dict(role="system", content=prompt),
        dict(
            role="user",
            content=json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False),
        ),
    ]
    wire = dict(response_format=response_format(packet, swapped))
    if controls["reasoning"] is not None:
        wire["reasoning"] = controls["reasoning"]
    byte_count = len(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
    ) + len(json.dumps(wire).encode())
    try:
        # A real run ID is a 64-character digest. Reserve 64 tokens beyond
        # the placeholder's count so its tokenization cannot break admission.
        tokens = input_token_bound(messages, wire, controls["tokenizer"], byte_cap) + 64
    except ValueError as error:
        if str(error) != "Annotation input exceeds independent byte cap":
            raise
        return dict(
            status="unavailable_packet_bytes", input_bytes=byte_count, input_token_bound=None
        )
    return dict(
        status=(
            "eligible"
            if packet.eligible and tokens <= token_cap
            else ("unavailable_packet_tokens" if tokens > token_cap else "unavailable_native_plan")
        ),
        input_bytes=byte_count,
        input_token_bound=tokens,
    )


def remaining(deadline, reserve=5):
    return max(
        0.0,
        min(float(deadline), float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf")))
        - time.time()
        - reserve,
    )


def native_plan(case, assignment, directory, deadline, case_seconds, *, rejection_precheck=True):
    directory = Path(directory)
    context = canonical_hash((case_to_dict(case), assignment))
    if type(rejection_precheck) is not bool:
        raise ValueError("rejection_precheck must be boolean")
    if not rejection_precheck:
        context = canonical_hash((context, "full-native-without-rejection-precheck/v1"))
    saved = directory / (context + ".json")
    if saved.exists():
        value = read(saved)
        if value["context"] != context:
            raise ValueError("Native checkpoint context changed")
        if not value["cleanup_complete"] or value["status"] == "error":
            raise RuntimeError("Retained native implementation/cleanup failure")
        return read_record(value["plan"]) if value.get("plan") else None, binding(saved)
    left = remaining(deadline)
    if left < 3:
        return None, None
    options = {} if rejection_precheck is True else {"rejection_precheck": rejection_precheck}
    result = bounded_call(
        _verified_plan,
        case_to_dict(case),
        assignment,
        timeout=min(case_seconds, left),
        memory_mb=8192,
        **options,
    )
    value = dict(
        context=context,
        status=result.status,
        detail=result.detail,
        cleanup_complete=result.cleanup_complete,
        resource_usage=dict(getattr(result, "resource_usage", ())),
        plan=(
            result.value.to_dict()
            if result.status == "complete" and result.value is not None
            else None
        ),
    )
    immutable(saved, value)
    if not result.cleanup_complete:
        raise RuntimeError("Native packet worker cleanup incomplete")
    if result.status == "error":
        raise RuntimeError("Native packet implementation error; inspect retained receipt")
    return read_record(value["plan"]) if value["plan"] else None, binding(saved)


def audit(prepared_path, output):
    prepared, output = read(prepared_path), Path(output)
    predecessor = bound(prepared["predecessor"])
    complete = bound(prepared["predecessor_completion"])
    terminal, outputs, _, _ = validate_completion(complete["audit_receipt"])
    report_path = Path(complete["report"]["path"])
    if outputs[str(report_path.relative_to(terminal["work"]))] != complete["report"]["sha256"]:
        raise ValueError("Prerequisite report lacks terminal output binding")
    report = bound(complete["report"])
    base = bound(predecessor["predecessor"])
    cases, _, _ = load_release(base["common_release"], base["audit_run"])
    by_id = {c.case_id: c for c in cases if c.split == "train"}
    template = bound(predecessor["annotation_templates"]["train"])
    source = prepared["controlled_source"]
    setting = bound(source)
    if (
        setting["settings"] != SETTINGS
        or setting["revision"] != REVISION
        or setting["scope"] != SCOPE
        or setting["glossary"] != GLOSSARY
    ):
        raise ValueError("Controlled setting changed after freeze")
    for key in ("constructor", "worker"):
        authenticate(setting[key])
    # Freeze all 128 settings before any native recheck, uniformly across arms.
    evidences = {}
    for case in by_id.values():
        check_time()
        path = output / "evidence" / (canonical_hash(case.case_id) + ".json")
        immutable(path, controlled_evidence(case, source))
        evidences[case.case_id] = binding(path)
    rows, shards = [], []
    for shard_index, reference in enumerate(report["shards"]):
        prerequisite = bound(reference)
        shard_rows = []
        for row in prerequisite["rows"]:
            check_time()
            case = by_id[row["case_id"]]
            if case.problem.content_hash != row["case_input_hash"]:
                raise ValueError("Frozen pair input changed")
            if row["pair"] is not None:
                actual = [
                    [
                        [obj.object_id, obj.candidates[i].candidate_id]
                        for obj, i in zip(case.problem.objects, assignment)
                    ]
                    for assignment in row["pair"]["assignments"]
                ]
                if actual != row["pair"]["candidate_ids"] or any(
                    len(a) != len(case.problem.objects) for a in row["pair"]["assignments"]
                ):
                    raise ValueError("Fixed pair assignments changed")
            row_path = output / "rows" / (canonical_hash(row["id"]) + ".json")
            dependencies = canonical_hash((row, prepared, evidences[case.case_id]))
            if row_path.exists():
                item = read(row_path)
                if item.get("preparation_dependencies") != dependencies:
                    raise ValueError("Committed packet row dependencies changed")
                for ref in [
                    *item["native_receipts"],
                    *([item["packet"]] if item["packet"] else []),
                ]:
                    authenticate(ref)
                shard_rows.append(item)
                rows.append(item)
                continue
            item = dict(
                row,
                preparation_dependencies=dependencies,
                comparison_id=row.get("original_id", row["id"]),
                evidence_contract=evidences[case.case_id],
                packet=None,
                native_receipts=[],
                paid_calls=0,
                replacement_allowed=False,
            )
            if row["pair"] is not None:
                plans = []
                # One fixed case balance across both plans; all native children share worker deadline.
                case_clock_path = output / "case-clocks" / (canonical_hash(case.case_id) + ".json")
                if not case_clock_path.exists():
                    admitted = time.time()
                    immutable(
                        case_clock_path,
                        dict(
                            case_id=case.case_id,
                            started_epoch=admitted,
                            deadline_epoch=min(
                                prepared["deadline_epoch"], admitted + prepared["case_seconds"]
                            ),
                            case_seconds=prepared["case_seconds"],
                            costs_reset=False,
                        ),
                    )
                case_clock = read(case_clock_path)
                pair_clock_path = (
                    output / "pair-clocks" / (canonical_hash(item["comparison_id"]) + ".json")
                )
                if not pair_clock_path.exists():
                    admitted = time.time()
                    immutable(
                        pair_clock_path,
                        dict(
                            comparison_id=item["comparison_id"],
                            started_epoch=admitted,
                            deadline_epoch=min(
                                case_clock["deadline_epoch"], admitted + prepared["pair_seconds"]
                            ),
                            pair_seconds=prepared["pair_seconds"],
                            costs_reset=False,
                        ),
                    )
                case_deadline = read(pair_clock_path)["deadline_epoch"]
                item["case_clock"] = binding(case_clock_path)
                item["pair_clock"] = binding(pair_clock_path)
                for assignment in row["pair"]["assignments"]:
                    plan, receipt = native_plan(
                        case, assignment, output / "native", case_deadline, prepared["plan_seconds"]
                    )
                    plans.append(plan)
                    if receipt:
                        item["native_receipts"].append(receipt)
                if any(p is None or not p.eligible for p in plans):
                    item["status"] = "unavailable_native_plan"
                else:
                    packet = packet_for(case, bound(evidences[case.case_id]), plans, template)
                    packet_path = output / "packets" / (packet.content_hash + ".json")
                    immutable(packet_path, packet.to_dict())
                    item.update(
                        packet=binding(packet_path),
                        **packet_admission(packet, template, row["swapped"]),
                    )
            item["primary_weak_label_eligible"] = item["status"] == "eligible"
            shard_rows.append(item)
            rows.append(item)
            immutable(row_path, item)
        path = output / f"train-{shard_index:02}.json"
        manifest = copy.deepcopy(template)
        manifest.update(
            authorized=False,
            deadline_epoch=prepared["deadline_epoch"],
            parent_splits={c.structural_parent: "train" for c in by_id.values()},
            slots=shard_rows,
            controlled_evidence_source=source,
            execution_gate="offline preparation; cumulative admission and closed-row runner integration required",
            seconds=min(28800, 92 * sum(r["status"] == "eligible" for r in shard_rows)),
        )
        immutable(path, manifest)
        shards.append(binding(path))
    originals = [r for r in rows if not r["swapped"]]
    if len(originals) != 256 or len(rows) != 308 or len(evidences) != 128:
        raise ValueError("Frozen TRAIN denominator changed")
    result = dict(
        schema="exact-repair/controlled-supervision-audit/v1",
        status="complete",
        prepared=binding(Path(prepared_path)),
        source=source,
        shards=shards,
        scheduled=len(rows),
        unique_comparisons=256,
        swaps=52,
        statuses=dict(Counter(r["status"] for r in rows)),
        original_statuses=dict(Counter(r["status"] for r in originals)),
        family_control_statuses=dict(
            Counter("/".join((r["family"], r["control"], r["status"])) for r in originals)
        ),
        evidence_cases=128,
        independent_semantic_evidence=False,
        cohort="generated_only",
        missing_real_coverage=True,
        hosted_calls=0,
        primary_fitting_admitted=False,
        test_outcomes_opened=False,
        request_limits=template["request_limits"],
        remaining_action="Review every row and complete cumulative packet admission before paid TRAIN; preserve unavailable rows and all prior reservations. Qualify combined loss and matched DEV reserve before fitting.",
    )
    immutable(output / "report.json", result)
    return result


def prepare(campaign, output, repository):
    campaign, output, repository = map(lambda p: Path(p).resolve(), (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    predecessor = registry["qualified_teacher_preparation"]["preparation"]
    completion_path = (
        campaign / "learning/qualified-teacher-shared-packets-clock-recovery-001/completed.json"
    )
    complete = bound(binding(completion_path))
    validate_completion(complete["audit_receipt"])
    base = bound(predecessor)
    selected = bound(base["selected_teacher"])
    review = bound(selected["review"])
    measured = [
        r["service_seconds"] for r in review["rows"] if r["profile"] == selected["selected_profile"]
    ]
    if len(measured) != 8:
        raise ValueError("Incomplete selected-teacher service observations")
    dev = bound(base["development_schedule"])
    reserve_by_model = {}
    for row in dev["rows"]:
        slot = row["selection_slot"]
        if slot["epoch"] == dev["final_epoch"]:
            key = str(slot["seed"]) + "/" + slot["supervision_condition"]
            reserve_by_model[key] = reserve_by_model.get(key, 0.0) + row["case_seconds"]
    dev.update(
        teacher_service_review=selected["review"],
        teacher_observed_seconds=dict(
            count=8, minimum=min(measured), maximum=max(measured), mean=sum(measured) / 8
        ),
        reserve_decision="Retain full 90-second request caps and 182/272-second verification/annotation reserves; eight calibration calls do not justify tighter tails.",
        final_pass_reserve_by_model=reserve_by_model,
        execution_authorized=False,
        schedule_status="matched_fixed_slots_pending_representative_runtime_and_common_endpoint",
    )
    immutable(output / "development-service-review.json", dev)
    runtime = source_identity()
    if runtime["dirty_hash"] != __import__("hashlib").sha256(b"").hexdigest():
        raise ValueError("Commit controlled setting and tested worker before freeze")
    ledger = read(campaign / "ledger.json")
    deadline = min(
        read(campaign / "campaign.json")["primary_freeze_epoch"],
        ledger["stages"]["acquisition"]["started_epoch"]
        + ledger["stage_limits"]["acquisition"]["elapsed_seconds"],
    )
    source_path = output / "controlled-setting.json"

    def committed_binding(relative):
        value = binding(repository / relative)
        value["path"] = str(campaign / "sources" / runtime["revision"] / "code" / relative)
        return value

    immutable(
        source_path,
        dict(
            schema="exact-repair/authored-controlled-setting/v1",
            revision=REVISION,
            settings=SETTINGS,
            glossary=GLOSSARY,
            scope=SCOPE,
            source_commit=runtime["revision"],
            constructor=committed_binding("tools/repair/corpus.py"),
            worker=committed_binding("tools/repair/grounded_supervision.py"),
            original_observations_unchanged=True,
            symbolic_targets_unchanged=True,
            independent_semantic_evidence=False,
            primary_semantic_readiness_claim=False,
        ),
    )
    prepared = dict(
        schema="exact-repair/controlled-supervision-preparation/v1",
        predecessor=predecessor,
        predecessor_completion=binding(completion_path),
        controlled_source=binding(source_path),
        deadline_epoch=deadline,
        plan_seconds=30,
        pair_seconds=65,
        case_seconds=130,
        source_commit=runtime["revision"],
        development_service_review=binding(output / "development-service-review.json"),
        hosted_calls=0,
        status="prepared_not_queued",
    )
    immutable(output / "prepared.json", prepared)
    job_id = "prepare-grounded-train-packets-001"
    job = dict(
        id=job_id,
        seconds=7200,
        slice_seconds=7200,
        cleanup_seconds=30,
        budget_stages=["acquisition", "learning"],
        stage="acquisition",
        priority=220,
        gpu_devices=[],
        resources=dict(cpus=3, gpus=0, memory_mb=20480, gres="none"),
        deadline_epoch=deadline,
        deadline_policy="defer",
        commands=[
            [
                "{python}",
                "-m",
                "pytest",
                "-q",
                "tests/repair_grounded_supervision_test.py",
                "--basetemp={work}/tests",
                "--junitxml={work}/tests.xml",
            ],
            [
                "{python}",
                "-m",
                "tools.repair.grounded_supervision",
                "audit",
                str(output / "prepared.json"),
                "{work}",
            ],
        ],
    )
    balance = _stage_remaining(copy.deepcopy(ledger), job, time.time())
    if job["seconds"] > 0.7 * balance:
        raise ValueError("Packet qualification exceeds 70 percent of remaining capacity")
    immutable(
        output / "admission.json",
        dict(
            remaining_seconds=balance,
            reserved_seconds=7200,
            scheduling_fraction=0.7,
            costs_reset=False,
        ),
    )
    base = bound(predecessor)
    spec = dict(
        repository=str(repository),
        campaign=str(campaign),
        allocation="14451",
        python="/home/pgcotovio/Exact-OM/.venv/bin/python",
        ledger=str(campaign / "ledger.json"),
        capacity=registry["capacity"],
        source_store=str(campaign / "sources"),
        protocol_source=base["protocols"][0]["path"],
        jobs=[job],
        input_files=[
            str(output / "prepared.json"),
            str(source_path),
            str(output / "admission.json"),
            str(output / "development-service-review.json"),
            predecessor["path"],
            str(completion_path),
        ],
        purpose="Complete fixed TRAIN packet qualification with authored controlled setting; no paid calls, fits or TEST",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(output / "batch-spec.json", campaign / "batches" / output.name)
    descriptor = prepare_dispatch(
        batch,
        job_id,
        campaign / "attempts" / job_id / "001",
        tmux_socket=campaign / "supervisor/tmux.sock",
        depends_on=["audit-qualified-teacher-shared-packets-clock-recovery-001"],
    )
    result = dict(
        preparation=binding(output / "prepared.json"),
        batch=binding(batch),
        descriptor=descriptor,
        status="prepared_not_queued",
    )
    immutable(output / "dispatch-prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.input, args.output, args.repository)
    else:
        audit(args.input, args.output)


if __name__ == "__main__":
    main()
