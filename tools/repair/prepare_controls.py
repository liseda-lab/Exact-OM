"""Freeze independent v3 common-inventory controls without opening reserved tests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.comparisons import freeze_controls
from exact.repair.learning import SemanticTargetSpec
from exact.repair.protocol import load_protocol_v3, training_projection_v3
from exact.repair.records import (
    RepairInputV3,
    canonical_hash,
    make_objective,
    read_record,
)
from exact.repair.study import StudyArmV2, StudyCaseV2, StudyOutcomeV3, load_schedule
from tools.repair.campaign_handoff import file_hash, label_dependencies
from tools.repair.prepare import load_preparation

ARMS = (
    StudyArmV2("no_repair", "no_repair", objective_key="uniform"),
    StudyArmV2("score_greedy_deletion", "deletion", "score_greedy", objective_key="uniform"),
    StudyArmV2("fixed_cost_deletion", "deletion", objective_key="uniform"),
    StudyArmV2("uniform", objective_key="uniform"),
    StudyArmV2("symbolic_retained_axioms", objective_key="symbolic"),
)
DEFERRED = "reserved_test_deferred_until_frozen_model_selection_and_B08_schedule"


def checked_preparation(source, protocol_path):
    """Require original v3 cache dependencies, provenance and complete non-test coverage."""
    protocol = training_projection_v3(load_protocol_v3(Path(protocol_path), for_execution=True))
    cases, caches, report = load_preparation(Path(source))
    if report["protocol_hash"] != canonical_hash(report["protocol"]):
        raise ValueError("Source preparation protocol identity is inconsistent")
    if label_dependencies(report["protocol"]) != label_dependencies(protocol):
        raise ValueError("Preparation dependencies differ from the control protocol")
    if any(c.schema_revision != "v3" or not isinstance(c.problem, RepairInputV3) for c in cases):
        raise ValueError("Controls require native v3 cases; historical inputs cannot be relabeled")
    selected = {c.case_id for c in cases if c.split in {"train", "development"}}
    if not selected or set(caches) != selected:
        raise ValueError("Require all non-test caches and no held-out or extraneous labels")
    profile = tuple(sorted(protocol["preferences"]["cost_weights"].items()))
    weights = protocol["teacher"]["family_weights"]
    for case in cases:
        if case.split == "test":
            continue
        cache = caches[case.case_id]
        expected = {
            "input": case.problem.content_hash,
            "patch": canonical_hash(case.problem.objects),
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(o.candidates for o in case.problem.objects)),
            "profile": canonical_hash(profile),
            "backend": canonical_hash(("pyhermit", version("pyhermit"), "python")),
            "teacher_weights": canonical_hash((weights["desired"], weights["unwanted"])),
            "semantic_target": SemanticTargetSpec(
                canonical_hash(case.probes), weights["desired"], weights["unwanted"]
            ).content_hash,
        }
        if (
            cache.schema != "exact-repair/teacher-cache/v3"
            or not cache.complete
            or any(dict(cache.hashes).get(k) != v for k, v in expected.items())
            or cache.candidate_counts != tuple(len(o.candidates) for o in case.problem.objects)
        ):
            raise ValueError("Unqualified teacher cache: " + case.case_id)
    return cases, caches, report, protocol, profile


def prepare_controls(source, protocol_path, output):
    """Save a deterministic schedule; teacher values never enter control objectives."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve the existing frozen control schedule")
    cases, caches, report, protocol, profile = checked_preparation(source, protocol_path)
    entries = []
    for case in cases:
        metadata = dict(
            case_id=case.case_id,
            cohort="generated",
            source_version="XR-2.1-T2-common-inventory",
            group_id=case.structural_parent,
            declared_split=case.split,
        )
        if case.split == "test":
            captured = StudyCaseV2(**metadata, availability="unavailable", detail=DEFERRED)
        else:
            # Keep tighter saved time limits and impose the campaign memory cap.
            # This operational projection does not change teacher dependencies.
            resources = protocol["resources"]
            saved = case.problem.budgets
            problem = replace(
                case.problem,
                budgets=replace(
                    saved,
                    total_seconds=min(saved.total_seconds, resources["case_wall_seconds"]),
                    verification_seconds=min(
                        saved.verification_seconds, resources["verification_seconds"]
                    ),
                    memory_mb=min(
                        saved.memory_mb or resources["case_rss_mb"], resources["case_rss_mb"]
                    ),
                ),
            )
            captured = freeze_controls(
                StudyCaseV2(
                    **metadata,
                    problem=problem,
                    objective=make_objective(
                        case.problem.objects,
                        profile=profile,
                        scale=protocol["objective"]["integer_scale"],
                    ),
                    capture_hash=case.problem.content_hash,
                )
            )
            variants = tuple(
                (
                    key,
                    replace(value, target_basis="diagnostic-" + key + "/v3"),
                )
                for key, value in captured.objective_variants
            )
            captured = replace(captured, objective_variants=variants)
        artifact = (
            output.parent / (output.stem + "_cases") / (canonical_hash(case.case_id) + ".json")
        )
        if artifact.exists():
            raise FileExistsError("Preserve existing control case artifacts")
        write_artifact(artifact, captured.to_dict())
        entries.append({**metadata, "artifact": str(artifact.relative_to(output.parent))})
    preparation = {
        "schema": "exact-repair/independent-controls/v1",
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "protocol_source": str(Path(protocol_path).resolve()),
        "label_dependency_hash": label_dependencies(protocol),
        "source_protocol_hash": report["protocol_hash"],
        "cache_hashes": {key: canonical_hash(cache) for key, cache in caches.items()},
        "original_label_seconds": report.get("label_seconds", 0),
        "original_label_cpu_seconds": report.get("label_cpu_seconds"),
        "case_counts": dict(Counter(c.split for c in cases)),
        "scheduled": len(cases) * len(ARMS),
        "eligible": sum(c.split != "test" for c in cases) * len(ARMS),
        "deferred": sum(c.split == "test" for c in cases) * len(ARMS),
        "labels_recomputed": False,
        "teacher_used_for_selection": False,
        "model_dependency": None,
        "test_evaluation": DEFERRED,
        "resource_policy": "retain tighter original time limits; apply protocol case RSS ceiling; cumulative batch allowance enforced separately",
        "scope": "common-inventory diagnostics; not generated-pool or strongest-symbolic evidence",
        "gates": "No G0-G2 or learning-efficiency claim is established by these controls",
    }
    schedule = {
        "schema": "exact-repair/study-schedule/v2",
        "seed": protocol["corpus"]["split_seed"],
        "cases": entries,
        "arms": [asdict(arm) for arm in ARMS],
        "preparation": preparation,
    }
    write_artifact(output, schedule)
    return preparation


def report_controls(schedule_path, results_path, output):
    """Keep every scheduled denominator and reuse labels only for post-selection diagnostics."""
    schedule_path, results_path = Path(schedule_path), Path(results_path)
    schedule = json.loads(schedule_path.read_text())
    preparation = schedule["preparation"]
    source = preparation["source"]
    if file_hash(source) != preparation["source_sha256"]:
        raise ValueError("Original preparation changed")
    _, caches, _, _, _ = checked_preparation(source, preparation["protocol_source"])
    if {key: canonical_hash(cache) for key, cache in caches.items()} != preparation["cache_hashes"]:
        raise ValueError("Original caches changed")
    cases, arms, _ = load_schedule(schedule_path)
    results = json.loads(results_path.read_text())
    indexed = {(r["case_id"], r["arm_id"]): r for r in results["rows"]}
    expected = {(c.case_id, a.arm_id) for c in cases for a in arms}
    if len(indexed) != len(results["rows"]) or set(indexed) - expected:
        raise ValueError("Unexpected or duplicate result rows")
    rows = []
    for case in cases:
        for arm in arms:
            row = dict(
                case_id=case.case_id,
                arm_id=arm.arm_id,
                split=case.declared_split,
                status="incomplete",
                logical_status="UNKNOWN",
                semantic_benefit=None,
                edit_cost=None,
                common_inventory_regret=None,
            )
            result = indexed.get((case.case_id, arm.arm_id))
            if result is not None:
                path = results_path.parent / result["artifact"]
                outcome = read_record(json.loads(path.read_text()))
                if not isinstance(outcome, StudyOutcomeV3) or (
                    outcome.case_hash != case.content_hash
                    or outcome.arm_hash != arm.content_hash
                    or outcome.content_hash != result["artifact_hash"]
                ):
                    raise ValueError("Result does not belong to the frozen control schedule")
                row.update(
                    status=outcome.status,
                    detail=outcome.detail,
                    logical_status=outcome.result.logical_status if outcome.result else "UNKNOWN",
                    resources=dict(outcome.resources),
                    outcome_sha256=file_hash(path),
                )
                if case.declared_split == "test":
                    if outcome.result is not None or outcome.common_assignment is not None:
                        raise ValueError("Reserved test case was evaluated")
                elif outcome.common_assignment is not None:
                    if (
                        outcome.result is None
                        or outcome.result.logical_status != "VERIFIED_FEASIBLE"
                        or outcome.result.verification is None
                        or not outcome.result.verification.authorizes
                    ):
                        raise ValueError("Selected assignment lacks completed verification")
                    cache = caches[case.case_id]
                    label = next(
                        (v for v in cache.labels if v.assignment == outcome.common_assignment), None
                    )
                    if label is not None and label.usable:
                        best = max(v.benefit - v.cost for v in cache.labels if v.usable)
                        row.update(
                            semantic_benefit=label.benefit,
                            edit_cost=label.cost,
                            common_inventory_regret=best - (label.benefit - label.cost),
                        )
            if case.declared_split == "test":
                row.update(status="deferred", detail=DEFERRED)
            rows.append(row)
    report = {
        "schema": "exact-repair/independent-control-report/v1",
        "schedule_sha256": file_hash(schedule_path),
        "source_preparation": preparation,
        "scheduled": len(rows),
        "recorded": len(indexed),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "rows": rows,
        "scientific_status": "exploratory_common_inventory_diagnostic",
        "primary_generated_pool_evaluation": "deferred_to_B08",
        "external_api_cost_usd": 0,
        "production_matcher": "deferred",
    }
    write_artifact(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source")
    prepare.add_argument("protocol")
    prepare.add_argument("output")
    report = commands.add_parser("report")
    report.add_argument("schedule")
    report.add_argument("results")
    report.add_argument("output")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_controls(args.source, args.protocol, args.output)
    else:
        result = report_controls(args.schedule, args.results, args.output)
    print(json.dumps({"scheduled": result["scheduled"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
