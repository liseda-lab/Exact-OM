"""Assemble the explicitly authorized TRAIN label correction without native replay.

The effective cache may replace only an old unknown, with a declared, terminally
authenticated correction. Original cases, inventories, attempts and caches remain
referenced. This is not an acquisition round or a general cache-union policy.
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.common_training import load_release
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_from_dict, publish_label_cache, read_label_cache
from tools.repair.shared_release import (
    aggregate, authenticate, bound, immutable, summarize_labels, validate_completion,
)


def substitute_unknowns(original, corrections):
    """Transport caches bind input only; the caller authenticates full declarations."""
    labels = {label.assignment: label for label in original.labels}
    seen = set()
    elapsed = original.elapsed_seconds
    for cache in corrections:
        if (cache.schema != original.schema
                or cache.candidate_counts != original.candidate_counts
                or dict(cache.hashes) != {"input": dict(original.hashes)["input"]}
                or len(cache.labels) != 1):
            raise ValueError("Correction transport dependencies changed")
        label = cache.labels[0]
        old = labels.get(label.assignment)
        if old is None or old.feasible is not None or label.assignment in seen:
            raise ValueError("Correction must replace one declared old unknown exactly once")
        seen.add(label.assignment)
        if label.feasible is not None:
            labels[label.assignment] = label
        elapsed += cache.elapsed_seconds
    return replace(original, labels=tuple(labels.values()), elapsed_seconds=elapsed,
                   stop_reason="authorized_source_corrected_unknowns")


def terminal_outputs(receipt):
    terminal, outputs, batch, job = validate_completion(receipt)
    authenticate(batch["source_manifest"])
    work = Path(terminal["work"])
    for path, digest in outputs.items():
        authenticate(dict(path=str(work / path), sha256=digest))
    return terminal, outputs, batch, job


def run(plan_path, output):
    import torch

    torch.set_num_threads(1)
    plan, output = read(plan_path), Path(output)
    if (plan.get("operation") != "authorized_unknown_label_correction"
            or plan.get("test_opened") is not False):
        raise ValueError("Expected explicit TRAIN correction plan")
    amendment = bound(plan["amendment"])
    previous = bound(amendment["previous_amendment"])
    if (amendment["schema"] != "exact-repair/train-label-technical-correction/v2"
            or not amendment["authorized"] or amendment["costs_reset"]
            or previous["prior_release"] != plan["prior_release"]):
        raise ValueError("Correction amendment changed")
    load_release(plan["prior_release"], previous["prior_audit"])
    base = bound(plan["prior_release"])
    prior_refs = {bound(ref)["case_id"]: ref for shard in base["shards"]
                  for ref in bound(shard)["rows"]}
    by_declaration, histories, expected, started = {}, [], set(), set()

    def collect(ref, terminal, outputs):
        path = authenticate(ref)
        if outputs.get(str(path.relative_to(terminal["work"]))) != ref["sha256"]:
            raise ValueError("Correction is not a terminal output")
        value = bound(ref)
        key = canonical_hash(value["declaration"])
        if key in by_declaration or not value["cleanup_complete"]:
            raise ValueError("Duplicate correction or incomplete cleanup")
        if value["status"] == "complete":
            artifact = value["artifact"]
            if outputs.get(str(Path(artifact["path"]).relative_to(terminal["work"]))) != artifact["sha256"]:
                raise ValueError("Correction cache is not a terminal output")
            authenticate(artifact)
        by_declaration[key] = (ref, value)

    terminal, outputs, _, _ = terminal_outputs(previous["qualification"])
    for ref in previous["reused_qualified_rows"]:
        collect(ref, terminal, outputs)
    for history in amendment["prior_runs"]:
        terminal, outputs, _, _ = terminal_outputs(history["run"])
        old_plan = bound(history["prior_plan"])
        expected.update(canonical_hash(d) for d in old_plan["rows"])
        for ref in history["started"]:
            path = authenticate(ref)
            if outputs.get(str(path.relative_to(terminal["work"]))) != ref["sha256"]:
                raise ValueError("Historical start is not a terminal output")
            started.add(canonical_hash(bound(ref)["declaration"]))
        for ref in history["completed"]:
            collect(ref, terminal, outputs)
        histories.append(history["run"])
    recovered = set()
    for receipt in plan["runs"]:
        terminal, outputs, batch, job = terminal_outputs(receipt)
        path = job["commands"][0][-2]
        frozen = dict(path=path, sha256=batch["frozen_files"][path])
        worker_plan = bound(frozen)
        if worker_plan["amendment"] != plan["amendment"]:
            raise ValueError("Recovery changed its amendment")
        report = bound(dict(path=str(Path(terminal["work"]) / "report.json"),
                            sha256=outputs["report.json"]))
        if report["plan"] != frozen or len(report["rows"]) != len(worker_plan["rows"]):
            raise ValueError("Recovery denominator changed")
        for declaration, ref in zip(worker_plan["rows"], report["rows"], strict=True):
            key = canonical_hash(declaration)
            if key in started or key in recovered or bound(ref)["declaration"] != declaration:
                raise ValueError("Recovery repeated started work or changed declaration")
            recovered.add(key)
            collect(ref, terminal, outputs)
    if recovered != expected - started or len(recovered) != amendment["pending_assignments"]:
        raise ValueError("Recovery omitted or added an assignment")
    grouped = defaultdict(list)
    for ref, value in by_declaration.values():
        old_ref = value["declaration"]["prior_row"]
        row = bound(old_ref)
        if row["split"] != "train" or prior_refs.get(row["case_id"]) != old_ref:
            raise ValueError("Correction changed frozen TRAIN membership")
        grouped[row["case_id"]].append((ref, value))
    rows = []
    for case_id, prior_ref in prior_refs.items():
        row = bound(prior_ref)
        corrections, receipts = [], []
        case = case_from_dict(bound(row["generated"])["case"]) if row.get("generated") else None
        for ref, value in grouped[case_id]:
            receipts.append(ref)
            if value["status"] != "complete":
                continue
            cache = read_label_cache(value["artifact"], Path(value["artifact"]["path"]).parent, case)
            if list(cache.labels[0].assignment) != value["declaration"]["assignment"]:
                raise ValueError("Correction cache changed the declared assignment")
            corrections.append(cache)
        if corrections:
            old = read_label_cache(row["cache"], Path(row["cache"]["path"]).parent, case)
            new = substitute_unknowns(old, corrections)
            row.update(cache=publish_label_cache(new, output / "caches" / canonical_hash(case_id)),
                       masks=summarize_labels(new, tuple(map(tuple, row["eligible_pairs"]))))
        row.update(prior_effective_row=prior_ref, correction_receipts=receipts)
        # Attempts and original timing fields are historical observations. Added
        # work is charged once through the bound recovery run ledger, not copied.
        rows.append(row)
    shards = []
    for start in range(0, len(rows), 8):
        refs = []
        for row in rows[start:start + 8]:
            path = output / "rows" / (canonical_hash(row["case_id"]) + ".json")
            immutable(path, row)
            refs.append(binding(path))
        path = output / "shards" / f"train-{start // 8:02}.json"
        immutable(path, dict(schema=base["schema"], shared_conditions=["symbolic", "symbolic_plus_llm"], rows=refs))
        shards.append(binding(path))
    coverage = dict(overall=aggregate(rows))
    for field in ("family", "control", "structural_parent"):
        coverage[field] = {key: aggregate([r for r in rows if r[field] == key])
                           for key in sorted({r[field] for r in rows})}
    immutable(output / "coverage.json", coverage)
    result = dict(base, operation=plan["operation"], shards=shards,
                  coverage=binding(output / "coverage.json"),
                  correction_predecessor=plan["prior_release"], correction_amendment=plan["amendment"],
                  correction_runs=plan["runs"], correction_histories=histories,
                  correction_qualification=previous["qualification"],
                  correction_plan=binding(plan_path), primary_training_admitted=False,
                  corrected_assignments=sum(v["status"] == "complete" and v.get("feasible") is not None
                                            for _, v in by_declaration.values()),
                  historical_failures_retained=True, native_calls=0)
    immutable(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
