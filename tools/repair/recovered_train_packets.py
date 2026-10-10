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

from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.common_training import load_release
from tools.repair.grounded_supervision import native_plan, packet_for
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, check_time, immutable

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
        old = bound(row["prior_native_row"])
        if old["pair"] is not None or old["packet"] is not None:
            raise ValueError("A previously bound pair cannot be replayed")
        plans = []
        for assignment in declaration["pair"]["assignments"]:
            native, receipt = native_plan(case, assignment, output / "native",
                                          read(pair_clock)["deadline_epoch"], plan["plan_seconds"])
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
              primary_training_admitted=False, test_opened=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
