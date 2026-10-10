"""Preselected, previously unavailable TRAIN pairs under a versioned task clarification.

No prior non-null pair or transmitted response can enter this worker. Native
failures stay terminal; the original 256-comparison schedule is not replaced.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import time

from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read
from tools.repair.common_training import load_release
from tools.repair.grounded_supervision import native_plan, packet_for
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, check_time, immutable, validate_completion

TASK_REVISION = "authored-retained-meaning-task/20261010-v2"
TASK = (
    "Assess retained meaning only under this explicitly authored fictional setting. "
    "D1-D3 are the supplied semantic evidence for that limited setting; independent "
    "real-world evidence is not required or claimed. Compare the complete reconstructed "
    "theories in plan_a.content and plan_b.content against those authored commitments. "
    "Expand the exact shared blocks and namespace dictionary in local_context first; "
    "they encode complete content, not missing content. Plan content is the object of "
    "comparison; cite the applicable packet.evidence keys for the meaning judgment. "
    "Do not invent plan evidence IDs or boolean entailment claims. Definitions with "
    "symbolic_value=null support semantic judgments, not checked symbolic claims. "
    "Use tie with supported equal scores when evidence supports equal retained meaning. "
    "Use abstain and unknown/null scores only when evidence is insufficient, ambiguous "
    "or contradictory for a criterion, not merely because neither plan is preferred. "
    "Do not infer external facts, reward named edits, or infer meaning quality from "
    "logical validity alone. Preserve the frozen criterion weights and score rubric."
)


def clarify(packet):
    if packet.split != "train":
        raise ValueError("Task amendment is TRAIN-only")
    return replace(packet, task=TASK)


def select_pairs(case, cache, old_rows):
    """Fill only never-bound pair slots; selection never reads LLM judgments."""
    if case.split != "train":
        raise ValueError("Pair recovery requires TRAIN membership")
    groups = defaultdict(list)
    for row in old_rows:
        if row["case_id"] != case.case_id:
            raise ValueError("Pair recovery changed case identity")
        groups[row["comparison_id"]].append(row)
    used = {tuple(sorted(tuple(a) for a in r["pair"]["assignments"]))
            for r in old_rows if r["pair"] is not None}

    def identity(a):
        return tuple((obj.object_id, obj.candidates[i].candidate_id)
                     for obj, i in zip(case.problem.objects, a, strict=True))

    assignments = sorted({label.assignment for label in cache.labels if label.usable},
                         key=lambda a: canonical_hash(identity(a))) if cache else []
    pairs = iter((a, b) for a, b in combinations(assignments, 2)
                 if tuple(sorted((a, b))) not in used)
    result = []
    for group in sorted(groups.values(), key=lambda g: g[0]["comparison_index"]):
        if any(row["pair"] is not None or row["packet"] is not None for row in group):
            continue
        if sorted(r["swapped"] for r in group) not in ([False], [False, True]):
            raise ValueError("Frozen original/swap schedule changed")
        pair = next(pairs, None)
        if pair is None:
            continue
        payload = dict(assignments=[list(a) for a in pair],
                       candidate_ids=[identity(a) for a in pair])
        result.extend(dict(row, pair=payload) for row in group)
    return result


def reuse_native(receipt):
    """Only a retained timeout gets one technical pass; no scientific replay."""
    value = bound(receipt)
    if not value["cleanup_complete"] or value["status"] == "error":
        raise ValueError("Unresolved predecessor native implementation/cleanup failure")
    if value["status"] == "timeout":
        return False, None
    return True, read_record(value["plan"]) if value.get("plan") else None


def recovery_history(plan):
    if "technical_recovery" not in plan:
        return None
    amendment = bound(plan["technical_recovery"])
    if (plan.get("rejection_precheck") is not False
            or amendment["same_cause_prior_attempts"] != 1
            or amendment["maximum_attempts"] != 2 or amendment["costs_reset"]
            or amendment["retry_completed_native"]):
        raise ValueError("Invalid one-pass native precheck recovery")
    qualification, qoutputs, _, _ = validate_completion(amendment["qualification"])
    qualified = bound(dict(path=str(Path(qualification["work"]) / "report.json"),
                           sha256=qoutputs["report.json"]))
    if not qualified["eligible"] or not qualified["full_native_policy_required"]:
        raise ValueError("Native correction has not qualified")
    qualified_native = qualified["native_receipt"]
    if qoutputs.get(str(Path(qualified_native["path"]).relative_to(qualification["work"]))) != qualified_native["sha256"]:
        raise ValueError("Qualified native receipt lacks terminal provenance")
    bound(qualified_native)
    previous_plan = bound(amendment["previous_plan"])
    if previous_plan.get("technical_recovery") or previous_plan.get("rejection_precheck") is False:
        raise ValueError("Same-cause technical pass already used")
    if {k: v for k, v in plan.items() if k not in {"technical_recovery", "rejection_precheck"}} != previous_plan:
        raise ValueError("Native recovery changed frozen rows or budgets")
    registered = amendment["prior_run"]
    attempt = Path(registered["completion_path"]).parent
    receipt = {key: binding(attempt / (key + ".json")) for key in ("completion", "step", "outputs")}
    receipt.update(batch=amendment["previous_batch"], dispatch_nonce=registered["dispatch_nonce"],
                   step_id=registered["step_id"], expected_status="complete")
    terminal, outputs, batch, _ = validate_completion(receipt)
    if batch["frozen_files"].get(amendment["previous_plan"]["path"]) != amendment["previous_plan"]["sha256"]:
        raise ValueError("Previous native plan was not frozen")
    work = Path(terminal["work"])
    report = bound(dict(path=str(work / "report.json"), sha256=outputs["report.json"]))
    if report["plan"] != amendment["previous_plan"]:
        raise ValueError("Previous native report changed")
    rows = {}
    for ref in report["rows"]:
        if outputs.get(str(Path(ref["path"]).relative_to(work))) != ref["sha256"]:
            raise ValueError("Previous row lacks terminal provenance")
        row = bound(ref)
        rows[row["id"]] = row
    if set(rows) != {row["id"] for row in bound(plan["rows"])["rows"]}:
        raise ValueError("Previous native denominator changed")
    return dict(receipt=receipt, work=work, outputs=outputs, rows=rows,
                qualified_native=qualified_native)


def run(plan_path, output):
    plan, output = read(plan_path), Path(output)
    amendment = bound(plan["task_amendment"])
    if (plan.get("test_opened") is not False or amendment["task_revision"] != TASK_REVISION
            or amendment["task"] != TASK or amendment["retry_prior_responses"] is not False):
        raise ValueError("Task amendment changed")
    cases, caches, _ = load_release(plan["release"], plan["release_audit"])
    by_id = {c.case_id: c for c in cases if c.split == "train"}
    template = bound(plan["annotation_template"])
    declarations = bound(plan["rows"])["rows"]
    history = recovery_history(plan)
    expected = []
    for case_id in sorted({r["case_id"] for r in declarations}):
        original_rows = bound(plan["original_rows"])["rows"]
        expected.extend(select_pairs(by_id[case_id], caches.get(case_id),
                                     [r for r in original_rows if r["case_id"] == case_id]))
    # Shards partition cases, so exact deterministic recomputation includes swaps.
    if canonical_hash(expected) != canonical_hash(declarations):
        raise ValueError("Preselected recovery pairs changed")
    rows = []
    for declaration in declarations:
        check_time()
        case = by_id[declaration["case_id"]]
        if case.problem.content_hash != declaration["case_input_hash"]:
            raise ValueError("Frozen case input changed")
        path = output / "rows" / (canonical_hash(declaration["id"]) + ".json")
        dependency = canonical_hash((binding(plan_path), declaration))
        if path.exists():
            row = read(path)
            if row["preparation_dependencies"] != dependency:
                raise ValueError("Saved native row changed")
            for ref in [*row["native_receipts"], *([row["packet"]] if row["packet"] else [])]:
                authenticate(ref)
            rows.append(row)
            continue
        case_clock = output / "case-clocks" / (canonical_hash(case.case_id) + ".json")
        if not case_clock.exists():
            immutable(case_clock, dict(started_epoch=time.time(), deadline_epoch=min(
                plan["deadline_epoch"], time.time() + plan["case_seconds"])))
        pair_clock = output / "pair-clocks" / (declaration["comparison_id"] + ".json")
        if not pair_clock.exists():
            immutable(pair_clock, dict(started_epoch=time.time(), deadline_epoch=min(
                read(case_clock)["deadline_epoch"], time.time() + plan["pair_seconds"])))
        row = dict(declaration, preparation_dependencies=dependency,
                   prior_native_row=plan["original_row_refs"][declaration["id"]],
                   task_amendment=plan["task_amendment"], packet=None, native_receipts=[],
                   case_clock=binding(case_clock), pair_clock=binding(pair_clock))
        if history:
            prior = history["rows"][declaration["id"]]
            row.update(prior_native_attempt=history["receipt"], attempt_index=2,
                       prior_case_clock=prior["case_clock"], prior_pair_clock=prior["pair_clock"],
                       cumulative_case_allowance_seconds=2 * plan["case_seconds"],
                       cumulative_pair_allowance_seconds=2 * plan["pair_seconds"],
                       costs_reset=False)
        old = bound(row["prior_native_row"])
        if old["pair"] is not None or old["packet"] is not None:
            raise ValueError("A previously bound pair cannot be replayed")
        plans = []
        for assignment in declaration["pair"]["assignments"]:
            if history:
                from tools.repair.prepare import case_to_dict

                old_context = canonical_hash((case_to_dict(case), assignment))
                qualified = bound(history["qualified_native"])
                context = canonical_hash((old_context, "full-native-without-rejection-precheck/v1"))
                if qualified["context"] == context:
                    saved = output / "native" / (context + ".json")
                    immutable(saved, qualified)
                    plans.append(read_record(qualified["plan"]))
                    row["native_receipts"].append(binding(saved))
                    continue
                prior_path = history["work"] / "native" / (old_context + ".json")
                digest = history["outputs"].get(str(prior_path.relative_to(history["work"])))
                if digest:
                    previous = dict(path=str(prior_path), sha256=digest)
                    reuse, native = reuse_native(previous)
                    if reuse:
                        plans.append(native)
                        row["native_receipts"].append(previous)
                        continue
            native, receipt = native_plan(case, assignment, output / "native",
                                          read(pair_clock)["deadline_epoch"], plan["plan_seconds"],
                                          rejection_precheck=plan.get("rejection_precheck", True))
            plans.append(native)
            if receipt:
                row["native_receipts"].append(receipt)
        if any(p is None or not p.eligible for p in plans):
            row["status"] = "unavailable_native_plan"
        else:
            packet = clarify(packet_for(case, bound(declaration["evidence_contract"]), plans, template))
            packet_path = output / "packets" / (packet.content_hash + ".json")
            immutable(packet_path, packet.to_dict())
            row.update(packet=binding(packet_path), status="complete_native_packet")
        row["primary_weak_label_eligible"] = False  # Input admission happens separately.
        immutable(path, row)
        rows.append(row)
    immutable(output / "report.json", dict(status="complete", plan=binding(plan_path),
              rows=[binding(output / "rows" / (canonical_hash(r["id"]) + ".json")) for r in rows],
              statuses=dict(Counter(r["status"] for r in rows)), hosted_calls=0,
              prior_native_receipt=history["receipt"] if history else None,
              primary_training_admitted=False, test_opened=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify-precheck", action="store_true")
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.qualify_precheck:
        from tools.repair.prepare import case_from_dict, case_to_dict

        plan = read(args.plan)
        old = bound(plan["old_timeout"])
        if old["status"] != "timeout" or not old["cleanup_complete"]:
            raise ValueError("Qualification requires the retained fixed timeout")
        case = case_from_dict(bound(plan["generated"])["case"])
        if case.split != "train":
            raise ValueError("Qualification requires a fixed TRAIN case")
        if old["context"] != canonical_hash((case_to_dict(case), plan["assignment"])):
            raise ValueError("Qualification changed the failed native assignment")
        started = time.time()
        native, receipt = native_plan(case, plan["assignment"], args.output / "native",
                                      min(plan["deadline_epoch"], started + 35), 30,
                                      rejection_precheck=False)
        immutable(args.output / "report.json", dict(
            plan=binding(args.plan), status="complete", native_receipt=receipt,
            eligible=bool(native and native.eligible), elapsed_seconds=time.time()-started,
            old_timeout_preserved=True, hosted_calls=0, full_native_policy_required=True))
    else:
        run(args.plan, args.output)


if __name__ == "__main__":
    main()
