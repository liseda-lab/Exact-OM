"""A predeclared paired conflict-incidence diagnostic, with native witnesses.

The four variants inherit one existing parent. They are never additional
independent test parents. Witnesses and semantic targets remain evaluator-only.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import time
from collections import Counter
from pathlib import Path

import pyowl_core as owl

from exact.repair.api import write_artifact
from exact.repair.candidates import make_candidate, replacement_cost_features
from exact.repair.learning import TeacherProbe
from exact.repair.records import (
    PolicyV2,
    RepairInputV2,
    RevisionObjectV2,
    canonical_hash,
    promote_input_v3,
    read_record,
    replace_inventory,
)
from tools.repair import fresh_evaluation as fresh
from tools.repair.batch import read, sha
from tools.repair.corpus import GeneratedCase, coherent_control
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair.expanded_profile import checked_checkpoint, checkpoint, parent_fingerprints
from tools.repair.prepare import case_to_dict

SCHEMA = "exact-repair/conflict-overlap/v1"
CONDITIONS = ("nonshared", "shared")
SUPPORTS = {"nonshared": ((0, 1), (2, 3)), "shared": ((0, 1), (0, 3))}


def construct(parent, condition):
    """Rewire one of four edges; equal objects, symbols, costs and soft probes.

    Project the original parent to one named anchor and replace its other theory
    with a controlled motif. This is an explicit diagnostic construction, not a
    claim that the parent's original semantics or complexity were preserved.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown conflict incidence condition")
    original = read_record(bound(parent["observable"]))
    anchor = sorted(original.policy.monitored_classes, key=owl.structural_hexdigest)[0]
    prefix = "urn:exact:conflict-incidence:" + canonical_hash(parent["group_id"]) + ":"
    x, y, a, b = [owl.Class(owl.IRI(prefix + n)) for n in ("X", "Y", "A", "B")]
    classes = (anchor, x, y, a, b)
    edges = ((anchor, x), (x, a), (anchor, y), (x if condition == "shared" else y, b))
    objects = []
    for i, (left, right) in enumerate(edges):
        name, axioms = "motif-" + str(i), (owl.SubClassOf(left, right),)
        objects.append(
            RevisionObjectV2(
                name,
                "mapping",
                axioms,
                (
                    make_candidate(name, axioms, ("keep",)),
                    make_candidate(
                        name,
                        (),
                        ("delete",),
                        cost_features=replacement_cost_features(
                            axioms, (), kind="mapping", authorship="generated"
                        ),
                    ),
                ),
                source="source",
                authorship="generated",
                source_entity=left,
                target_entity=right,
            )
        )
    fixed = (owl.DisjointClasses((anchor, a)), owl.DisjointClasses((anchor, b)))
    problem = promote_input_v3(
        RepairInputV2(
            fixed,
            tuple(objects),
            PolicyV2(classes),
            source_identity="controlled-conflict-incidence-source",
            target_identity="controlled-conflict-incidence-target",
            matcher_identity="synthetic-controlled-no-matcher",
            evidence=tuple((o.object_id, {"score": 0.8}) for o in objects),
            source_axioms=fixed,
        )
    )
    intended = (0, 1, 0, 1)
    case = GeneratedCase(
        parent["base_case_id"] + ":incidence:" + condition,
        parent["structural_parent"],
        "conflict_incidence",
        "test",
        problem,
        (
            TeacherProbe("retain-left-prefix", owl.SubClassOf(anchor, x), "path"),
            TeacherProbe("retain-right-prefix", owl.SubClassOf(anchor, y), "path"),
        ),
        intended,
        20261003,
        False,
        intended_theory=(
            *fixed,
            *(ax for o, i in zip(objects, intended) for ax in o.candidates[i].axioms),
        ),
        variation=(("condition", condition),),
        schema_revision="v3",
    )
    clean = coherent_control(case)
    # An absent clean mapping is not an editable object with an empty original
    # relation: endpoint retrieval requires a complete asserted mapping bundle.
    # Keep the intended clean theory, dropping only its two absent objects.
    retained = [i for i, obj in enumerate(clean.problem.objects) if obj.original_axioms]
    clean = dataclasses.replace(
        clean,
        problem=replace_inventory(clean.problem, tuple(clean.problem.objects[i] for i in retained)),
        intended_assignment=tuple(clean.intended_assignment[i] for i in retained),
    )
    return case, clean


def exposure_audit(cases, parent, split):
    """Union all variants and their ancestry before checking cross-split aliases."""
    inventory = bound(split["profile_inventory"])
    fingerprints = set(parent["fingerprints"])
    for case in cases:
        fingerprints.update(parent_fingerprints(case))
    rows = [dict(r, split="historically_exposed") for r in inventory["exposed"]]
    rows += list(inventory["selected"])
    # Transitive closure matters: one alias can connect another variant/split.
    matches, changed = {}, True
    while changed:
        changed = False
        for row in rows:
            if fingerprints.intersection(row["fingerprints"]) and row["key"] not in matches:
                matches[row["key"]] = row
                fingerprints.update(row["fingerprints"])
                changed = True
    return dict(
        source_parent=parent["group_id"],
        inherited_split=parent["split"],
        ancestry_union_fingerprints=sorted(fingerprints),
        matching_parents=list(matches.values()),
        cross_split_or_historical=any(r["split"] != "fresh_evaluation" for r in matches.values()),
        independent_parent_count=1,
        heldout_claim=False,
        policy="All four variants and any structural aliases form one ancestry group. "
        "Diagnostic motif projection only; no held-out transfer or population inference.",
    )


def prepare(source_schedule, output, *, amendment=None):
    """Freeze design from observable structure only; do not open any result rows."""
    base, output = bound(source_schedule), Path(output)
    parents = sorted(
        (
            c
            for c in base["cases"]
            if c["family"] == "overlap"
            and c.get("control") == "corrupted"
            and c.get("condition") == "baseline"
            and c["status"] == "materialized"
        ),
        key=lambda c: (c["group_id"], c["case_id"]),
    )
    if not parents:
        raise ValueError("Predeclared overlap ancestry is unavailable")
    parent = parents[0]
    cases, items = [], []
    for condition in CONDITIONS:
        for case in construct(parent, condition):
            cases.append(case)
            key = canonical_hash((case.case_id, condition))
            items.append(
                dict(
                    case_id=case.case_id,
                    base_case_id=parent["base_case_id"],
                    group_id=parent["group_id"],
                    structural_parent=case.structural_parent,
                    source_parent=parent["group_id"],
                    family=case.family,
                    family_exposure="diagnostic_ancestry_inherited_no_transfer_claim",
                    split=parent["split"],
                    control=case.control,
                    condition=condition,
                    status="materialized",
                    input_hash=case.problem.content_hash,
                    observable=immutable(
                        output / "observable" / (key + ".json"), case.problem.to_dict()
                    ),
                    evaluator=immutable(output / "evaluator" / (key + ".json"), case_to_dict(case)),
                    expected_minimal_supports=(
                        SUPPORTS[condition] if case.control == "corrupted" else ()
                    ),
                    fingerprints=parent_fingerprints(case),
                )
            )
    audit = exposure_audit(cases, parent, bound(base["split_schedule"]))
    schedule = {
        k: copy.deepcopy(base[k])
        for k in (
            "schema",
            "program",
            "authorization",
            "corpus_completion",
            "split_schedule",
            "arms",
            "control_definition",
            "symbolic_limitation",
            "model_selection_source",
        )
    }
    schedule.update(
        study_kind="paired_conflict_incidence_diagnostic",
        cases=items,
        rows=fresh.make_rows(items, schedule["arms"]),
        source_schedule=source_schedule,
        parent_source={k: parent[k] for k in ("observable", "group_id", "fingerprints", "split")},
        structural_audit=immutable(output / "structural-audit.json", audit),
        planned_cases=4,
        planned_parent_groups=1,
        planned_model_rows=24,
        planned_control_rows=12,
        scheduled_rows=36,
        seed=20261003,
        followup="xr21-expanded-robustness-overlap-report-001",
        study_complete=False,
        generation_seconds=60,
        test_outcomes_opened=False,
        api_spend_usd=0,
        design="Four mapping objects, five classes, two immutable disjointness axioms. "
        "Only edge 3 changes X->B to Y->B; shared supports {0,1}/{0,3}, "
        "nonshared {0,1}/{2,3}. Intended repair deletes edges 1 and 3. "
        "Clean controls observe that intended theory as two nonempty mapping objects; "
        "they are coherence controls, not mapping-count-matched corrupt comparisons. "
        "Same prefix probes and edit-cost rules.",
        claim_scope="Single paired controlled construction; no G0-G2, learning-efficiency, "
        "held-out transfer, strongest-symbolic superiority or population claim.",
        selection_policy="Lexicographically first declared fresh overlap ancestry; no outcome selection.",
        witness_policy="Enumerate all 16 keep/delete subsets per corrupted motif and verify "
        "exact minimal conflicts natively; clean controls require full coherence. "
        "Any incomplete or mismatching evidence retains all corresponding arm rows as unavailable.",
    )
    if len(schedule["arms"]) != 9 or len(schedule["rows"]) != 36:
        raise ValueError("Six frozen models and three controls are required")
    if amendment is not None:
        bound(amendment)
        schedule["preexecution_design_amendment"] = amendment
    immutable(output / "schedule.json", schedule)
    return schedule


def native_witness(item, directory):
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    problem = read_record(bound(item["observable"]))
    verifier = OwlVerifier("hermit", backend="python")
    indices = tuple(range(len(problem.objects)))
    subsets = (
        [indices]
        if item["control"] == "coherent"
        else [
            values for n in range(len(indices) + 1) for values in itertools.combinations(indices, n)
        ]
    )
    results, conflicts = [], []
    for subset in subsets:
        axioms = (
            *problem.fixed_axioms,
            *(a for i in subset for a in problem.objects[i].original_axioms),
        )
        report = verifier.check_theory(
            snapshot_from_axioms(axioms), problem.policy.monitored_classes
        )
        proof = immutable(
            Path(directory) / ("subset-" + ("-".join(map(str, subset)) or "empty") + ".json"),
            report.to_dict(),
        )
        qualified = (
            report.complete
            and report.support.complete_imports
            and report.logical_status in {"VERIFIED_FEASIBLE", "VERIFIED_INFEASIBLE"}
        )
        results.append(
            dict(subset=subset, qualified=qualified, status=report.logical_status, proof=proof)
        )
        if qualified and report.logical_status == "VERIFIED_INFEASIBLE":
            conflicts.append(set(subset))
    minimal = sorted(tuple(sorted(s)) for s in conflicts if not any(t < s for t in conflicts))
    expected = sorted(tuple(s) for s in item["expected_minimal_supports"])
    return dict(
        case_id=item["case_id"],
        input_hash=problem.content_hash,
        checks=results,
        actual_minimal_supports=minimal,
        expected_minimal_supports=expected,
        qualified=all(r["qualified"] for r in results) and minimal == expected,
    )


def witness_identity(schedule_path):
    from exact.repair.study import runtime_manifest

    return canonical_hash((sha(schedule_path), sha(__file__), runtime_manifest()))


def run_witness(schedule_path, output):
    from exact.repair.workers import bounded_call

    schedule, output = read(schedule_path), Path(output)
    identity, rows = witness_identity(schedule_path), []
    output.mkdir(parents=True, exist_ok=True)
    for item in schedule["cases"]:
        key = canonical_hash(item)
        row_path, guard = output / "rows" / (key + ".json"), output / "inflight" / (key + ".json")
        row_identity = canonical_hash((identity, item))
        saved = checked_checkpoint(row_path, row_identity)
        if saved is None:
            if guard.exists():
                raise RuntimeError(
                    "Witness inflight ownership requires reconciliation; no silent repeat"
                )
            checkpoint(guard, row_identity, case_id=item["case_id"], started_epoch=time.time())
            started = time.monotonic()
            call = bounded_call(
                native_witness,
                item,
                str(output / "proofs" / key),
                timeout=300,
                cpu_seconds=600,
                memory_mb=8192,
            )
            fresh.ensure_cleanup(call)
            saved = checkpoint(
                row_path,
                row_identity,
                case_id=item["case_id"],
                status=call.status,
                detail=call.detail,
                cleanup_complete=call.cleanup_complete,
                elapsed_seconds=time.monotonic() - started,
                resources=dict(call.resource_usage),
                result=call.value if call.status == "complete" else None,
            )
            guard.unlink()
        if guard.exists() or not saved["cleanup_complete"]:
            raise RuntimeError("Witness ownership or cleanup unresolved")
        if saved.get("result"):
            for check in saved["result"]["checks"]:
                bound(check["proof"])
        rows.append(binding(row_path))
    return checkpoint(
        output / "report.json",
        identity,
        schema=SCHEMA,
        schedule=binding(schedule_path),
        status="complete",
        rows=rows,
        qualified=sum(bool((bound(r).get("result") or {}).get("qualified")) for r in rows),
        scheduled=4,
        study_complete=False,
    )


def qualified_schedule(schedule_path, witness_path):
    schedule = read(schedule_path)
    report = checked_checkpoint(Path(witness_path), witness_identity(schedule_path))
    if report is None or report["status"] != "complete" or len(report["rows"]) != 4:
        raise ValueError("Missing complete witness collection")
    if report["schedule"] != binding(schedule_path):
        raise ValueError("Witness belongs to another schedule")
    results = {bound(r)["case_id"]: bound(r) for r in report["rows"]}
    if set(results) != {c["case_id"] for c in schedule["cases"]}:
        raise ValueError("Witness denominator differs")
    for item in schedule["cases"]:
        saved = results[item["case_id"]]
        if saved["identity"] != canonical_hash((witness_identity(schedule_path), item)):
            raise ValueError("Witness case identity differs")
        value = saved.get("result")
        if value:
            for check in value["checks"]:
                bound(check["proof"])
        if not value or not value["qualified"]:
            item["status"] = "unavailable_native_conflict_witness"
    schedule["witness_report"] = binding(witness_path)
    return schedule


def evaluate(schedule_path, witness_path, output, start, stop):
    effective = Path(output) / "qualified-schedule.json"
    immutable(effective, qualified_schedule(schedule_path, witness_path))
    return fresh.run(effective, Path(output) / "evaluation", start, stop)


def summarize(schedule_path, witness_path, reports, output):
    expected, rows, seen = qualified_schedule(schedule_path, witness_path), [], set()
    for path in reports:
        report = read(path)
        if report["status"] != "complete" or bound(report["schedule"]) != expected:
            raise ValueError("Evaluation completion/schedule differs")
        report_identity = canonical_hash(
            (report["schedule"]["sha256"], sha(fresh.__file__), report["runtime"])
        )
        checked_checkpoint(Path(path), report_identity)
        for ref in report["rows"]:
            saved = bound(ref)
            fresh.validate_payloads(saved)
            row = saved["row"]
            checked_checkpoint(Path(ref["path"]), canonical_hash((report_identity, row)))
            if row not in expected["rows"] or row["id"] in seen:
                raise ValueError("Unexpected/duplicate overlap evaluation row")
            seen.add(row["id"])
            item = expected["cases"][row["case_index"]]
            result = bound(saved["result"]) if saved.get("result") else {}
            rows.append(
                dict(
                    row_id=row["id"],
                    arm_id=row["arm_id"],
                    condition=item["condition"],
                    control=item["control"],
                    status=saved["status"],
                    semantic_benefit=result.get("semantic_benefit"),
                    logical_status=result.get("logical_status", "UNKNOWN"),
                    elapsed_seconds=saved["elapsed_seconds"],
                    resources=saved["resources"],
                    receipt=ref,
                )
            )
    if seen != {r["id"] for r in expected["rows"]}:
        raise ValueError("Incomplete overlap evaluation denominator")
    pairs = []
    for arm in expected["arms"]:
        for control in ("corrupted", "coherent"):
            pair = {
                r["condition"]: r
                for r in rows
                if r["arm_id"] == arm["id"] and r["control"] == control
            }
            known = all(
                pair[c]["logical_status"] == "VERIFIED_FEASIBLE"
                and pair[c]["semantic_benefit"] is not None
                for c in CONDITIONS
            )
            pairs.append(
                dict(
                    arm_id=arm["id"],
                    control=control,
                    available=known,
                    shared_minus_nonshared_benefit=(
                        (pair["shared"]["semantic_benefit"] - pair["nonshared"]["semantic_benefit"])
                        if known
                        else None
                    ),
                    shared_minus_nonshared_seconds=(
                        pair["shared"]["elapsed_seconds"] - pair["nonshared"]["elapsed_seconds"]
                    ),
                    rows={c: pair[c]["row_id"] for c in CONDITIONS},
                )
            )
    result = dict(
        schema=SCHEMA,
        status="complete",
        schedule=binding(schedule_path),
        witness_report=binding(witness_path),
        scheduled=36,
        recorded=len(rows),
        rows=rows,
        pairs=pairs,
        evaluation_reports=[binding(p) for p in reports],
        outcomes=dict(Counter(r["status"] for r in rows)),
        claim_scope=expected["claim_scope"],
        independent_parent_count=1,
        uncertainty="One diagnostic parent; no bootstrap interval or population inference",
        study_complete=True,
        api_spend_usd=0,
    )
    write_artifact(Path(output) / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("witness", "evaluate", "report"))
    parser.add_argument("schedule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--witness", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--reports", nargs="*", type=Path)
    args = parser.parse_args()
    if args.stage == "witness":
        run_witness(args.schedule, args.output)
    elif args.stage == "evaluate":
        evaluate(args.schedule, args.witness, args.output, args.start, args.stop)
    else:
        summarize(args.schedule, args.witness, args.reports, args.output)


if __name__ == "__main__":
    main()
