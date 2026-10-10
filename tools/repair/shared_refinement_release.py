"""Freeze the final shared TRAIN release, preserving both acquisition rounds."""

import argparse
from pathlib import Path

from exact.repair.learning import TeacherCache
from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_from_dict, read_label_cache, publish_label_cache
from tools.repair.shared_release import (
    SCHEMA,
    aggregate,
    audit_case,
    authenticate,
    bound,
    immutable,
    validate_completion,
)


def union_caches(original, refinement):
    """Append compatible labels; prior unknowns cannot be upgraded by replay."""
    if (original.schema, original.candidate_counts, dict(original.hashes)) != (
        refinement.schema,
        refinement.candidate_counts,
        dict(refinement.hashes),
    ):
        raise ValueError("Refinement cache scientific dependencies changed")
    labels = {row.assignment: row for row in original.labels}
    for row in refinement.labels:
        if row.assignment in labels and row != labels[row.assignment]:
            raise ValueError("Refinement changed a prior committed label")
        labels.setdefault(row.assignment, row)
    return TeacherCache(
        original.candidate_counts,
        tuple(labels.values()),
        False,
        "shared_round0_plus_round1",
        original.hashes,
        original.elapsed_seconds + refinement.elapsed_seconds,
        original.schema,
    )


def run(plan_path, output):
    import torch
    from tools.repair.shared_release import summarize_labels

    torch.set_num_threads(1)
    plan, output = read(plan_path), Path(output)
    if (
        plan.get("schema") != SCHEMA
        or plan.get("operation") != "merge_refinement"
        or plan.get("heldout_outcomes_opened") is not False
    ):
        raise ValueError("Invalid final common release plan")
    base = bound(plan["prior_release"])
    if base["status"] != "complete" or base.get("acquisition_closed"):
        raise ValueError("Exactly one second acquisition round is allowed")
    rows, references = {}, {}
    for shard in base["shards"]:
        for ref in bound(shard)["rows"]:
            row = bound(ref)
            if row["case_id"] in rows:
                raise ValueError("Duplicate baseline case")
            rows[row["case_id"]], references[row["case_id"]] = row, ref
    if len(rows) != 128 or sum(r["scheduled_attempts"] for r in rows.values()) != 2048:
        raise ValueError("Baseline denominator changed")
    processed, run_receipts = set(), []
    for run in plan["runs"]:
        completion, outputs, batch, job = validate_completion(run)
        if completion["status"] != "complete":
            raise ValueError("Failed refinement job requires explicit recovery lineage")
        authenticate(batch["source_manifest"])
        acquire_path = job["commands"][0][-2]
        acquire_ref = dict(path=acquire_path, sha256=batch["frozen_files"][acquire_path])
        acquire = bound(acquire_ref)
        if acquire["round_id"] != "1" or acquire["prior_release"] != plan["prior_release"]:
            raise ValueError("Refinement belongs to another round or base release")
        acquire["_path"] = acquire_path
        work = Path(completion["work"])
        report = bound(dict(path=str(work / "report.json"), sha256=outputs["report.json"]))
        if report["plan"] != acquire_ref or len(report["rows"]) != len(acquire["case_ids"]):
            raise ValueError("Refinement report differs from its frozen case schedule")
        for ref in report["rows"]:
            key = bound(ref)["case_id"]
            if key not in acquire["case_ids"] or key in processed or key not in rows:
                raise ValueError("Duplicate or out-of-scope refinement case")
            processed.add(key)
            original = rows[key]
            if acquire["prior_rows"][key] != references[key]:
                raise ValueError("Refinement baseline row changed")
            second = audit_case(original["input"], ref, acquire, outputs, work)
            second_path = output / "refinement" / (canonical_hash(key) + ".json")
            immutable(second_path, second)
            merged = dict(
                original,
                round_id="0+1",
                prior_round=references[key],
                refinement=binding(second_path),
                scheduled_attempts=32,
                attempts=[dict(s, round_id="0") for s in original["attempts"]]
                + [dict(s, round_id="1") for s in second["attempts"]],
                elapsed_seconds=original["elapsed_seconds"] + second["elapsed_seconds"],
                native_cpu_seconds=original["native_cpu_seconds"] + second["native_cpu_seconds"],
                peak_rss_bytes=max(original["peak_rss_bytes"], second["peak_rss_bytes"]),
                process_status="complete" if second["process_status"] == "complete" else "partial",
                scientific_status=(
                    "collected" if second.get("cache") else "round1_" + second["scientific_status"]
                ),
                round_statuses={"0": original["process_status"], "1": second["process_status"]},
            )
            if second.get("cache"):
                case = case_from_dict(bound(original["generated"])["case"])
                if original["generated"]["sha256"] != second["generated"]["sha256"]:
                    raise ValueError("Refinement changed generated inventory bytes")
                old = read_label_cache(
                    original["cache"], Path(original["cache"]["path"]).parent, case
                )
                new = read_label_cache(second["cache"], Path(second["cache"]["path"]).parent, case)
                combined = union_caches(old, new)
                artifact = publish_label_cache(combined, output / "caches" / canonical_hash(key))
                merged.update(
                    cache=artifact,
                    masks=summarize_labels(
                        combined, tuple(tuple(p) for p in original["eligible_pairs"])
                    ),
                    reused_assignment_labels=len(old.labels)
                    + len(new.labels)
                    - len(combined.labels),
                    added_assignment_labels=len(combined.labels) - len(old.labels),
                )
            rows[key] = merged
        run_receipts.append(run)
    if processed != set(plan["expected_refinement_case_ids"]):
        raise ValueError("Final refinement denominator changed")
    ordered = list(rows.values())
    shards = []
    for start in range(0, 128, 8):
        refs = []
        for row in ordered[start : start + 8]:
            path = output / "rows" / (canonical_hash(row["case_id"]) + ".json")
            immutable(path, row)
            refs.append(binding(path))
        path = output / "shards" / (f"train-{start // 8:02d}.json")
        immutable(
            path,
            dict(schema=SCHEMA, shared_conditions=["symbolic", "symbolic_plus_llm"], rows=refs),
        )
        shards.append(binding(path))
    coverage = dict(overall=aggregate(ordered))
    for field in ("family", "control", "structural_parent"):
        coverage[field] = {
            key: aggregate([r for r in ordered if r[field] == key])
            for key in sorted({r[field] for r in ordered})
        }
    coverage["family_control"] = {
        family
        + "/"
        + control: aggregate(
            [r for r in ordered if r["family"] == family and r["control"] == control]
        )
        for family in sorted({r["family"] for r in ordered})
        for control in sorted({r["control"] for r in ordered})
    }
    immutable(output / "coverage.json", coverage)
    authenticate(base["development"])
    immutable(
        output / "report.json",
        dict(
            schema=SCHEMA,
            status="complete",
            operation="merge_refinement",
            plan=binding(plan_path),
            prior_release=plan["prior_release"],
            cohort="generated_only",
            expected_train_cases=128,
            expected_development_cases=32,
            expected_assignment_slots=2048 + 16 * len(processed),
            round0_assignment_slots=2048,
            round1_assignment_slots=16 * len(processed),
            shards=shards,
            development=base["development"],
            coverage=binding(output / "coverage.json"),
            acquisition_closed=True,
            maximum_rounds=2,
            refinement_runs=run_receipts,
            costs="Each original/refinement attempt charged once in the existing ledger; reused labels incur no native recheck",
            common=base["common"],
            conference=base["conference"],
            missing_real_coverage=True,
            fit_execution_authorized=False,
            heldout_outcomes_opened=False,
            hosted_calls=0,
            optimizer_updates=0,
            interaction_learning_claim=False,
            nonzero_quartet_data_available=bool(coverage["overall"]["useful_quartet_parents"]),
        ),
    )
    return read(output / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
