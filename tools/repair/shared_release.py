"""Authenticate generated TRAIN receipts and freeze a common, reference-only release.

No reasoner, acquisition, hosted call, optimizer or held-out evaluator is invoked.
Unavailable cases and interrupted slot tails remain in the original denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from exact.repair.api import write_artifact
from exact.repair.learning import SemanticTargetSpec, interaction_loss
from exact.repair.records import canonical_hash, canonical_json
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_from_dict, read_label_cache

SCHEMA = "exact-repair/common-generated-shared-release/v1"


def check_time():
    if time.time() + 2 >= float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf")):
        raise TimeoutError("Shared-release audit deadline; prior row checkpoints retained")


def authenticate(reference):
    check_time()
    path = Path(reference["path"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            check_time()
            digest.update(data)
    if digest.hexdigest() != reference["sha256"]:
        raise ValueError("Shared release dependency changed: " + str(path))
    return path


def bound(reference):
    value = read(authenticate(reference))
    if "identity" in value and "content_hash" in value:
        if canonical_hash({k: v for k, v in value.items() if k != "content_hash"}) != value["content_hash"]:
            raise ValueError("Invalid checkpoint content hash")
    return value


def immutable(path, value):
    value = json.loads(canonical_json(value))
    if path.exists():
        if read(path) != value:
            raise ValueError("Immutable shared release already differs: " + str(path))
    else:
        write_artifact(path, value)


def reconcile_slots(attempts, terminal_status, count=16):
    """Do not infer that an uncommitted interruption tail was never transmitted."""
    if len(attempts) > count or [r.get("order") for r in attempts] != list(range(len(attempts))):
        raise ValueError("Assignment slot order/denominator changed")
    return [dict(row) for row in attempts] + [
        dict(order=i, status="not_committed_" + terminal_status, assignment=None)
        for i in range(len(attempts), count)
    ]


def validate_completion(run):
    completion, step = bound(run["completion"]), bound(run["step"])
    for record in (completion, step):
        if record.get("dispatch_nonce") != run["dispatch_nonce"] or record.get("step_id") != run["step_id"]:
            raise ValueError("Completion nonce/Slurm step mismatch")
    if completion["status"] != run["expected_status"]:
        raise ValueError("Unexpected terminal acquisition status")
    if completion["exit_code"] != (0 if completion["status"] == "complete" else 1):
        raise ValueError("Terminal acquisition exit status mismatch")
    outputs = bound(run["outputs"])
    batch = bound(run["batch"])
    if Path(run["batch"]["path"]) != Path(completion["batch"]):
        raise ValueError("Completion names another frozen batch")
    job = next(row for row in batch["jobs"] if row["id"] == completion["job_id"])
    return completion, outputs, batch, job


def summarize_labels(cache, pairs):
    import torch

    # Reuse the authoritative feasible, same-background quartet definition.
    quartets = interaction_loss(
        torch.zeros(len(cache.labels)), cache.labels, eligible_pairs=pairs, max_quartets=100000
    )
    values = [label.benefit for label in cache.labels if label.usable]
    risks = [float(label.feasible is False) for label in cache.labels if label.feasible is not None]
    return dict(
        labels=len(cache.labels),
        symbolic_value=sum(label.usable for label in cache.labels),
        symbolic_rank_pairs=sum(a != b for i, a in enumerate(values) for b in values[i + 1:]),
        risk=len(risks),
        risk_positive=sum(risks),
        unknown_policy=sum(label.feasible is None for label in cache.labels),
        unknown_semantics=sum(label.feasible is True and label.benefit is None for label in cache.labels),
        proposal_candidate_labels=len(values),
        proposal_exact=cache.complete,
        proposal_eligibility="candidate targets only; circuit reachability and missing mass checked at fit",
        pair=quartets["eligible"],
        quartet_positive=quartets["positive"],
        quartet_negative=quartets["negative"],
        quartet_zero=quartets["zero"],
        semantic_values=values,
        quartet_targets=list(quartets["targets"]),
        weak_anchor=0,
        weak_comparison=0,
    )


def audit_case(declaration, reference, plan, outputs, work):
    receipt = bound(reference)
    if outputs.get(str(Path(reference["path"]).relative_to(work))) != reference["sha256"]:
        raise ValueError("Case receipt is not a bound terminal output")
    if any(receipt[k] != declaration[v] for k, v in (
        ("case_id", "case_id"), ("family", "family"), ("control", "control"), ("parent", "structural_parent")
    )) or receipt["scheduled_attempts"] != 16:
        raise ValueError("Case identity/denominator changed")
    if receipt.get("cleanup_complete") is not True:
        raise ValueError("Case cleanup incomplete")
    for item in receipt.get("native_evidence", []):
        path = authenticate(item)
        if path.name == "call.json" and read(path).get("cleanup_complete") is False:
            raise ValueError("Native child cleanup incomplete")
    state = bound(receipt["partial_state"]) if receipt.get("partial_state") else {}
    for item in state.get("calls", []):
        if bound(item).get("cleanup_complete") is not True:
            raise ValueError("Nested acquisition cleanup incomplete")
    original = case_from_dict(bound(declaration["evaluator"]))
    if original.problem.content_hash != declaration["input_hash"] or original.split != "train":
        raise ValueError("TRAIN input identity changed")
    result = receipt.get("result") or {}
    partial = state.get("collection_state", {})
    collection = bound(result["collection"]) if result.get("collection") else None
    attempts = (collection or partial).get("attempts", result.get("attempts", []))
    slots = reconcile_slots(attempts, receipt["status"])
    record = dict(
        case_id=receipt["case_id"], family=receipt["family"], control=receipt["control"],
        structural_parent=receipt["parent"], split="train", round_id="0",
        input=declaration, completion=reference, plan=binding(plan["_path"]),
        process_status=receipt["status"], scientific_status=result.get("status", "no_committed_result"),
        scheduled_attempts=16, attempts=slots, masks=None,
        partial_labels_retained_not_admitted=len(partial.get("labels", [])) if not collection else 0,
        partial_state=receipt.get("partial_state"),
        elapsed_seconds=receipt.get("resources", {}).get("wall_seconds", 0.0),
        native_cpu_seconds=state.get("cpu_seconds", 0.0),
        peak_rss_bytes=receipt.get("resources", {}).get("peak_sampled_tree_rss_bytes", 0),
        native_evidence_count=len(receipt.get("native_evidence", [])),
        cleanup_complete=True,
    )
    if collection:
        if receipt["status"] != "complete" or result.get("status") != "collected":
            raise ValueError("Partial failure cannot be admitted as complete collection")
        generated = bound(result["generated"])
        case = case_from_dict(generated["case"])
        cache = read_label_cache(result["cache"], Path(reference["path"]).parent / "cache", case)
        if canonical_hash(asdict(cache)) != canonical_hash(collection["cache"]):
            raise ValueError("Published cache and collection disagree")
        if (case.case_id, case.structural_parent, case.split, case.probes) != (
            original.case_id, original.structural_parent, "train", original.probes
        ):
            raise ValueError("Generated case changed identity, split or target")
        dependencies = collection["collection_dependencies"]
        if canonical_hash(dependencies) != collection["collection_identity"]:
            raise ValueError("Collection dependency hash changed")
        if (collection["case_id"], collection["parent_group_id"], collection["split"], collection["round_id"]) != (
            case.case_id, case.structural_parent, "train", "0"
        ) or collection["requested"] != 16 or len(attempts) != 16:
            raise ValueError("Collection scope changed")
        protocol = bound(plan["protocol"])
        target = SemanticTargetSpec(canonical_hash(case.probes),
            protocol["teacher"]["family_weights"]["desired"], protocol["teacher"]["family_weights"]["unwanted"])
        expected = dict(input=case.problem.content_hash, patch=canonical_hash(case.problem.objects),
            policy=case.problem.policy.content_hash, query=canonical_hash(case.probes),
            inventory=canonical_hash(tuple(o.candidates for o in case.problem.objects)),
            profile=canonical_hash(tuple(sorted(protocol["objective"]["edit_weights"].items()))),
            semantic_target=target.content_hash)
        hashes = dict(cache.hashes)
        if any(hashes.get(k) != v for k, v in expected.items()) or hashes != dependencies["hashes"]:
            raise ValueError("Cache input/policy/query/inventory/target dependency changed")
        pairs = tuple(tuple(pair) for pair in state.get("pairs", []))
        for slot in slots:
            if "label_index" in slot and tuple(slot["assignment"]) != cache.labels[slot["label_index"]].assignment:
                raise ValueError("Attempt points to another assignment label")
        record.update(generated=result["generated"], cache=result["cache"], collection=result["collection"],
            collection_identity=collection["collection_identity"], collection_dependencies=dependencies,
            generator_model_hash=generated["model_hash"], eligible_pairs=pairs,
            masks=summarize_labels(cache, pairs), cache_complete=cache.complete,
            quartet_transport_affected=bool(pairs) and any(
                slot.get("stratum") == "quartet" and slot.get("assignment") is None and slot["status"] == "unavailable"
                for slot in slots))
    return record


def aggregate(rows):
    parents = {r["structural_parent"] for r in rows}
    masks = [r["masks"] for r in rows if r["masks"] is not None]
    values = [v for m in masks for v in m["semantic_values"]]
    targets = [v for m in masks for v in m["quartet_targets"]]
    variance = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs) if xs else None
    risk = sum(m["risk"] for m in masks)
    positives = sum(m["risk_positive"] for m in masks)
    return dict(
        cases=len(rows), parents=len(parents), scheduled_slots=16 * len(rows),
        slot_statuses=dict(Counter(s["status"] for r in rows for s in r["attempts"])),
        scientific_statuses=dict(Counter(r["scientific_status"] for r in rows)),
        process_statuses=dict(Counter(r["process_status"] for r in rows)),
        usable_cases=sum(bool(r["masks"] and r["masks"]["symbolic_value"]) for r in rows),
        usable_parents=len({r["structural_parent"] for r in rows if r["masks"] and r["masks"]["symbolic_value"]}),
        useful_quartet_parents=len({r["structural_parent"] for r in rows if r["masks"] and (
            r["masks"]["quartet_positive"] or r["masks"]["quartet_negative"])}),
        counts={k: sum(m[k] for m in masks) for k in (
            "labels", "symbolic_value", "symbolic_rank_pairs", "risk", "unknown_policy", "unknown_semantics",
            "proposal_candidate_labels", "pair", "quartet_positive", "quartet_negative", "quartet_zero", "weak_anchor", "weak_comparison")},
        semantic_target_variance=variance(values), quartet_target_variance=variance(targets),
        constant_risk_baseline=dict(source="TRAIN", eligible=risk, probability=positives / risk if risk else None),
        timeout_inclusive_case_seconds=sum(r["elapsed_seconds"] for r in rows),
        max_case_seconds=max((r["elapsed_seconds"] for r in rows), default=0),
        peak_rss_bytes=max((r["peak_rss_bytes"] for r in rows), default=0),
        quartet_transport_affected_cases=sum(r.get("quartet_transport_affected", False) for r in rows),
    )


def run(plan_path, output):
    import torch

    torch.set_num_threads(1)
    plan, output = read(plan_path), Path(output)
    if plan.get("schema") != SCHEMA or plan.get("heldout_outcomes_opened") is not False:
        raise ValueError("Invalid shared release plan")
    index = bound(plan["inputs"])
    for name, split, count in (("training", "train", 128), ("development", "development", 32)):
        rows = index[name + "_rows"]
        if len(rows) != count or len({r["case_id"] for r in rows}) != count or any(r["split"] != split for r in rows):
            raise ValueError("Frozen TRAIN/DEV denominator changed")
        if canonical_hash(sorted(rows, key=lambda r: r["case_id"])) != canonical_hash(sorted(
            bound(index[name + "_manifest"])["rows"], key=lambda r: r["case_id"])):
            raise ValueError("Input manifest differs from index")
    if {r["structural_parent"] for r in index["training_rows"]} & {r["structural_parent"] for r in index["development_rows"]}:
        raise ValueError("TRAIN/DEV ancestry overlap")
    expected = {r["case_id"]: r for r in index["training_rows"]}
    recovery = bound(plan["recovery"])
    completed, audit_runs, common = {}, [], None
    for item in plan["runs"]:
        check_time()
        completion, outputs, batch, job = validate_completion(item)
        # Bind current runner source separately from unchanged scientific preparation.
        authenticate(batch["source_manifest"])
        commands = job["commands"]
        acquisition_path = commands[0][-2]
        acquisition = read(acquisition_path)
        if batch["frozen_files"].get(acquisition_path) != binding(acquisition_path)["sha256"]:
            raise ValueError("Acquisition plan is not frozen in its batch")
        shared = {k: acquisition[k] for k in ("inputs", "protocol", "checkpoint", "generator_source", "seed", "round_id", "plan_quotas", "target_revision")}
        if common is None:
            common = shared
        if shared != common or acquisition["inputs"] != plan["inputs"] or acquisition["round_id"] != "0":
            raise ValueError("Acquisition shards do not share scientific dependencies")
        for key in ("checkpoint", "generator_source", "protocol"):
            authenticate(acquisition[key])
        acquisition["_path"] = acquisition_path
        work = Path(completion["work"])
        if completion["status"] == "failed":
            if item["completion"] != recovery["original_completion"] or item["dispatch_nonce"] != recovery["original_dispatch_nonce"]:
                raise ValueError("Unapproved failed acquisition lineage")
            references = recovery["retained_rows"]
        else:
            report_ref = dict(path=str(work / "report.json"), sha256=outputs["report.json"])
            report = bound(report_ref)
            if report["status"] != "complete" or report["expected_rows"] != len(report["rows"]):
                raise ValueError("Acquisition report incomplete")
            references = report["rows"]
            if report["plan"] != binding(acquisition_path):
                raise ValueError("Report belongs to another acquisition plan")
        for reference in references:
            key = bound(reference)["case_id"]
            if key not in expected or key not in acquisition["case_ids"] or key in completed:
                raise ValueError("Duplicate, omitted or out-of-split case receipt")
            completed[key] = audit_case(expected[key], reference, acquisition, outputs, work)
            immutable(output / "rows" / (canonical_hash(key) + ".json"), completed[key])
            write_artifact(output / "progress.json", dict(completed_rows=len(completed), expected_rows=128))
        audit_runs.append(dict(**item, elapsed_seconds=completion["elapsed_seconds"],
            peak_rss_mb=completion["peak_rss_mb"], source=batch["source_manifest"]))
    if set(completed) != set(expected):
        raise ValueError("Shared release must contain all 128 TRAIN cases exactly once")
    rows = [completed[r["case_id"]] for r in index["training_rows"]]
    development = []
    for row in index["development_rows"]:
        # Only authenticate input bytes: no DEV outcome or evaluation is opened.
        authenticate(row["evaluator"])
        authenticate(row["observable"])
        development.append(dict(**row, cache=None, outcome="not_evaluated", included_in_fitting=False))
    immutable(output / "development-inputs.json", dict(schema=SCHEMA, rows=development, expected_rows=32))
    shards = []
    for start in range(0, 128, 8):
        path = output / "shards" / (f"train-{start // 8:02d}.json")
        immutable(path, dict(schema=SCHEMA, shared_conditions=["symbolic", "symbolic_plus_llm"],
            rows=[binding(output / "rows" / (canonical_hash(r["case_id"]) + ".json")) for r in rows[start:start + 8]]))
        shards.append(binding(path))
    summary = dict(overall=aggregate(rows))
    for field in ("family", "control", "structural_parent"):
        summary[field] = {key: aggregate([r for r in rows if r[field] == key]) for key in sorted({r[field] for r in rows})}
    summary["family_control"] = {family + "/" + control: aggregate([
        r for r in rows if r["family"] == family and r["control"] == control])
        for family in sorted({r["family"] for r in rows}) for control in sorted({r["control"] for r in rows})}
    immutable(output / "coverage.json", summary)
    immutable(output / "report.json", dict(schema=SCHEMA, status="complete", plan=binding(plan_path),
        cohort="generated_only", expected_train_cases=128, expected_assignment_slots=2048,
        expected_development_cases=32, heldout_outcomes_opened=False, hosted_calls=0, optimizer_updates=0,
        fit_execution_authorized=False, shards=shards, development=binding(output / "development-inputs.json"),
        coverage=binding(output / "coverage.json"), runs=audit_runs, common=common,
        conference=plan["conference"], missing_real_coverage=True,
        nonzero_quartet_data_available=bool(summary["overall"]["useful_quartet_parents"]),
        interaction_learning_claim=False,
        paid_teacher_status="unqualified_separate_panel_amendment_pending"))
    return read(output / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
