"""Small frozen-row corrective study runner with qualified complete-plan controls.

Selection receives deployment inputs and explicitly supplied observable queries.
Generated-case reference probes are opened only after the selected plan commits.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.candidates import (
    make_candidate,
    deduplicate_candidates,
    replacement_cost_features,
)
from exact.repair.kernel import repair, materialize
from exact.repair.learning import SemanticTargetSpec, OwlTeacherOracle
from exact.repair.owl import OwlVerifier, snapshot_from_axioms
from exact.repair.records import (
    canonical_hash,
    read_record,
    replace_inventory,
    make_objective,
    candidate_cost,
)
from exact.repair.workers import bounded_call, SUPERVISION_GRACE_SECONDS
from tools.repair.prepare import case_from_dict, _probe

SCHEMA = "exact-repair/corrective-study/v1"
PUBLIC_INPUT_SCHEMA = "exact-repair/public-study-input/v1"


def bound(value):
    if isinstance(value, dict) and {"path", "sha256"} <= set(value):
        raw = Path(value["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("frozen study artifact checksum mismatch")
        return json.loads(raw)
    return value


def elementary_pool(problem):
    """Native deletion never invokes retrieval, rich grammar, or a neural model."""
    objects = []
    for obj in problem.objects:
        original = tuple(
            c for c in obj.candidates if c.axioms == obj.original_axioms and "keep" in c.action_tags
        )
        if not original:
            original = (make_candidate(obj.object_id, obj.original_axioms, ("keep",)),)
        candidates = original
        if obj.eligible and not obj.locked:
            candidates += (
                make_candidate(
                    obj.object_id,
                    (),
                    ("delete",),
                    cost_features=replacement_cost_features(
                        obj.original_axioms, (), kind=obj.kind, authorship=obj.authorship
                    ),
                ),
            )
        objects.append(dataclasses.replace(obj, candidates=deduplicate_candidates(candidates)))
    return replace_inventory(problem, tuple(objects))


def _query_value(problem, assignment, probes):
    axioms, active = materialize(problem, assignment)
    snapshot = snapshot_from_axioms(axioms)
    verifier = OwlVerifier("auto", backend="auto")
    target = SemanticTargetSpec(canonical_hash(probes))
    return target.evaluate(OwlTeacherOracle(verifier, snapshot, probes), probes).benefit


def _whole_plan_search(problem, probes, profile, directory, seconds):
    """Evaluate nonadditive query utility externally; never encode it as unary credit."""
    if not probes:
        raise ValueError("complete-plan observable-query search requires a declared query basis")
    started = time.monotonic()
    best = None
    attempted = qualified = scored = 0
    first_verified = None
    exhaustive = True
    pool_size = 1
    for obj in problem.objects:
        pool_size *= len(obj.candidates)
    for assignment in itertools.product(*(range(len(obj.candidates)) for obj in problem.objects)):
        left = seconds - (time.monotonic() - started)
        if left <= 0.25:
            exhaustive = False
            break
        attempted += 1
        selected_objects = tuple(
            dataclasses.replace(obj, candidates=(obj.candidates[index],))
            for obj, index in zip(problem.objects, assignment)
        )
        singleton = replace_inventory(problem, selected_objects)
        singleton = dataclasses.replace(
            singleton,
            budgets=dataclasses.replace(
                singleton.budgets,
                total_seconds=max(0.001, min(left / 2, problem.budgets.verification_seconds * 3)),
            ),
        )
        objective = make_objective(singleton.objects, profile=profile)
        result = repair(
            singleton,
            objective,
            diagnose=True,
            preserve_verified_input=False,
            initial_assignment=tuple(0 for _ in singleton.objects),
        )
        if (
            result.logical_status == "VERIFIED_INFEASIBLE"
            or result.search_status == "NO_FEASIBLE_IN_POOL"
        ):
            qualified += 1
            continue
        if result.verification is None or not result.verification.authorizes:
            exhaustive = False
            continue
        qualified += 1
        if first_verified is None:
            first_verified = time.monotonic() - started
        left = seconds - (time.monotonic() - started)
        label = bounded_call(
            _query_value,
            singleton,
            result.assignment,
            probes,
            timeout=max(0.001, min(left, problem.budgets.verification_seconds)),
            memory_mb=problem.budgets.memory_mb,
        )
        if not label.cleanup_complete:
            exhaustive = False
            break
        if label.status != "complete" or label.value is None:
            exhaustive = False
            continue
        scored += 1
        cost = sum(
            candidate_cost(obj, candidate, profile)
            for obj, candidate in zip(singleton.objects, result.selected)
        )
        utility = float(label.value) - cost
        candidate = dict(
            utility=utility,
            query_value=float(label.value),
            edit_cost=cost,
            assignment_ids=[c.candidate_id for c in result.selected],
            input=singleton.to_dict(),
            objective=objective.to_dict(),
            result=result.to_dict(),
        )
        write_artifact(Path(directory) / f"observable-plan-{attempted:06d}.json", candidate)
        if best is None or utility > best["utility"]:
            best = candidate
            write_artifact(Path(directory) / "observable-incumbent.json", best)
    return dict(
        best=best,
        first_verified_seconds=first_verified,
        attempted=attempted,
        qualified=qualified,
        scored=scored,
        total_assignments=pool_size,
        search_status=(
            "OPTIMAL_IN_SHARED_POOL"
            if exhaustive and attempted == pool_size
            else "BEST_CHECKED_NO_OPTIMALITY_CLAIM"
        ),
        query_basis_hash=canonical_hash(probes),
    )


def evaluate_row(case_record, settings, directory):
    """Actual inference/control entry point; returned artifacts bind their exact pool."""
    from exact.repair.pipeline import (
        bounded_freeze_checkpoint,
        repair_neural_round,
        staged_verified_repair,
    )
    from tools.repair.evaluate_campaign import generation_options
    from tools.repair.fresh_evaluation import generate_control

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["EXACT_REPAIR_PHASE_JOURNAL"] = str(directory / "phases")
    case_record = bound(case_record)
    # Do not deserialize evaluator targets before prediction.
    public_input = case_record.get("schema") == PUBLIC_INPUT_SCHEMA
    if public_input:
        public = {**case_record["metadata"], "problem": case_record["input"]}
        required = {"case_id", "family", "structural_parent", "split", "control"}
        if not required <= public.keys():
            raise ValueError(
                "public study inputs require explicit case/parent/split/scope metadata"
            )
    else:
        public = case_record["case"]
    problem = read_record(public["problem"])
    protocol = bound(settings["protocol"])
    resources = protocol["resources"]
    seconds = float(settings.get("seconds", resources["case_wall_seconds"]))
    if os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH"):
        seconds = min(seconds, float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]) - time.time())
    if seconds <= 0:
        return dict(
            status="resource_limited", logical_status="UNKNOWN", detail="case deadline exhausted"
        )
    started = time.monotonic()
    problem = dataclasses.replace(
        problem,
        budgets=dataclasses.replace(
            problem.budgets,
            total_seconds=seconds,
            verification_seconds=resources["verification_seconds"],
            memory_mb=float(settings.get("memory_mb", resources["case_rss_mb"])),
        ),
    )
    arm = settings["arm"]
    profile = tuple(sorted(protocol["objective"]["edit_weights"].items()))
    summary = dict(
        status="unresolved",
        case_id=public["case_id"],
        family=public["family"],
        structural_parent=public["structural_parent"],
        control=public.get("control", "corrupted"),
        split=public["split"],
        theory_scope=settings.get("theory_scope", "complete_asserted_input"),
        input_hash=problem.content_hash,
        arm=arm,
        semantic_benefit=None,
        logical_status="UNKNOWN",
        source_revision=settings.get("source_revision", "unspecified"),
        hardware=dict(host=platform.node(), machine=platform.machine()),
    )
    result = selected_problem = objective = None
    selection_offset = 0.0
    staged_first_verified = None
    if settings.get("frozen_inventory"):
        shared = read_record(bound(settings["frozen_inventory"]))
        if (
            shared.fixed_axioms != problem.fixed_axioms
            or [(o.object_id, o.original_axioms) for o in shared.objects]
            != [(o.object_id, o.original_axioms) for o in problem.objects]
            or (shared.policy.required, shared.policy.prohibited, shared.policy.exceptions)
            != (problem.policy.required, problem.policy.prohibited, problem.policy.exceptions)
        ):
            raise ValueError("shared pool changed the captured input or hard policy")
        problem = dataclasses.replace(shared, budgets=problem.budgets)
    if arm == "native_deletion":
        selected_problem = elementary_pool(problem)
        objective = make_objective(
            selected_problem.objects, profile=profile, scale=protocol["objective"]["integer_scale"]
        )
        selection_offset = time.monotonic() - started
        result = repair(selected_problem, objective, diagnose=True)
        summary["generation_status"] = "ELEMENTARY_DIRECT_NO_RICH_GENERATION"
    elif arm == "learned":
        checkpoint = settings["checkpoint"]
        options = generation_options(protocol, directory / "compiler-cache")
        options.update(settings.get("generation_overrides", {}))
        schedule = settings.get(
            "schedule", protocol["generation"].get("execution_schedule", "one_stage")
        )
        repair_settings = dict(
            shortlist_size=protocol["selection"].get("shortlist_size", 1),
            utility_window=protocol["selection"].get("utility_window", 0),
            ledger_path=str(directory / "search-ledger.json"),
        )
        reserve = min(problem.budgets.verification_seconds, seconds / 5)
        if schedule == "staged_verified_repair":
            stage_offset = time.monotonic() - started
            outcome = staged_verified_repair(
                problem,
                checkpoint_path=checkpoint["path"],
                checkpoint_sha256=checkpoint["sha256"],
                total_seconds=max(0.001, seconds - reserve),
                elementary_seconds=protocol["generation"].get("elementary_seconds", 30),
                generation_options=options,
                repair_options=repair_settings,
            )
            summary.update(generation_status=outcome.generation_status, stages=list(outcome.stages))
            staged_times = [
                r["first_verified_seconds"]
                for r in outcome.stages
                if r.get("first_verified_seconds") is not None
            ]
            staged_first_verified = stage_offset + min(staged_times) if staged_times else None
            if outcome.frozen is not None:
                selected_problem, objective = outcome.frozen.problem, outcome.frozen.objective
                result = outcome.result
        elif schedule == "one_stage":
            generated = bounded_freeze_checkpoint(
                problem,
                checkpoint["path"],
                checkpoint_sha256=checkpoint["sha256"],
                seconds=min(resources["generation_seconds"], max(0.001, seconds - reserve)),
                memory_mb=problem.budgets.memory_mb,
                **options,
            )
            summary.update(
                generation_status=generated.status, cleanup_complete=generated.cleanup_complete
            )
            if generated.status == "complete" and generated.cleanup_complete:
                frozen = generated.value
                left = seconds - (time.monotonic() - started) - reserve
                if left > 0:
                    frozen = dataclasses.replace(
                        frozen,
                        problem=dataclasses.replace(
                            frozen.problem,
                            budgets=dataclasses.replace(frozen.problem.budgets, total_seconds=left),
                        ),
                    )
                    selection_offset = time.monotonic() - started
                    result = repair_neural_round(frozen, **repair_settings)
                    selected_problem, objective = frozen.problem, frozen.objective
        else:
            raise ValueError("undeclared repair schedule")
    elif arm in {"support_retention_surrogate", "complete_plan_observable_query"}:
        probes = tuple(_probe(value) for value in settings.get("observable_queries", ()))
        if not probes:
            raise ValueError("symbolic semantic control requires explicit observable query records")
        if not settings.get("frozen_inventory"):
            generated = bounded_call(
                generate_control,
                problem,
                "uniform",
                protocol,
                str(directory),
                timeout=min(resources["generation_seconds"], seconds / 3),
                memory_mb=problem.budgets.memory_mb,
            )
            if generated.status != "complete" or not generated.cleanup_complete:
                return dict(
                    summary,
                    status="generation_" + generated.status,
                    cleanup_complete=generated.cleanup_complete,
                )
            problem = generated.value.problem
        left = seconds - (time.monotonic() - started)
        if arm == "complete_plan_observable_query":
            selection_offset = time.monotonic() - started
            search = _whole_plan_search(problem, probes, profile, directory, max(0.001, left * 0.8))
            staged_first_verified = (
                selection_offset + search["first_verified_seconds"]
                if search.get("first_verified_seconds") is not None
                else None
            )
            summary["observable_search"] = {k: v for k, v in search.items() if k != "best"}
            if search["best"] is not None:
                selected_problem = read_record(search["best"]["input"])
                objective = read_record(search["best"]["objective"])
                result = read_record(search["best"]["result"])
        else:
            # Explicit local support surrogate, not nonadditive entailment credit.
            values = tuple(
                tuple(
                    sum((1 if p.desired else -1) for p in probes if p.axiom in c.axioms)
                    / len(probes)
                    for c in obj.candidates
                )
                for obj in problem.objects
            )
            objective = make_objective(
                problem.objects,
                values,
                profile=profile,
                scale=protocol["objective"]["integer_scale"],
            )
            selected_problem = dataclasses.replace(
                problem,
                budgets=dataclasses.replace(problem.budgets, total_seconds=max(0.001, left * 0.8)),
            )
            selection_offset = time.monotonic() - started
            result = repair(selected_problem, objective, diagnose=True)
            summary["surrogate_contract"] = "local_asserted_query_support/v1"
    else:
        raise ValueError("unsupported corrective study arm")
    if result is not None:
        write_artifact(
            directory / "selection.json",
            dict(
                input=selected_problem.to_dict(),
                objective=objective.to_dict(),
                result=result.to_dict(),
            ),
        )
        summary.update(
            status="evaluated",
            logical_status=result.logical_status,
            search_status=result.search_status,
            selected_ids=[c.candidate_id for c in result.selected],
            first_verified_seconds=(
                staged_first_verified
                if staged_first_verified is not None
                else (
                    selection_offset + result.first_verified_seconds
                    if result.first_verified_seconds is not None
                    else None
                )
            ),
            first_verified_timing_scope="row_worker_including_generation",
            checks=result.checks,
            master_solves=result.solves,
            stage_seconds=dict(result.stage_seconds),
            verification_scope=result.verification_scope,
            selected_input_hash=selected_problem.content_hash,
            initial_status=next(
                (r.verdict for name, r in result.baseline if name == "alignment"), "UNKNOWN"
            ),
        )
        if "observable_search" in summary:
            summary["search_status"] = summary["observable_search"]["search_status"]
        if public_input:
            # Actual captured ontologies/alignments have no invented clean parent
            # or generated-reference truth. Independent evaluators attach later.
            summary["semantic_status"] = "independent_evaluation_not_provided"
        elif result.verification is not None and result.verification.authorizes:
            # Reference probes first enter the process after the selected plan is committed.
            case = case_from_dict(case_record)
            remaining = seconds - (time.monotonic() - started)
            if remaining > 0:
                evaluated = bounded_call(
                    _query_value,
                    selected_problem,
                    result.assignment,
                    case.probes,
                    timeout=min(remaining, problem.budgets.verification_seconds),
                    memory_mb=problem.budgets.memory_mb,
                )
                summary.update(
                    semantic_status=evaluated.status,
                    semantic_benefit=evaluated.value if evaluated.status == "complete" else None,
                    cleanup_complete=evaluated.cleanup_complete,
                )
    summary["wall_seconds"] = time.monotonic() - started
    write_artifact(directory / "row-summary.json", summary)
    return summary


def run(manifest_path, output, start=0, stop=None, deadline_epoch=None):
    if deadline_epoch is None and os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH"):
        deadline_epoch = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"])
    # The allocation deadline includes descendant cleanup and final receipts.
    cleanup_reserve = max(1.0, 2 * SUPERVISION_GRACE_SECONDS)
    manifest = bound(
        {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        }
    )
    if manifest.get("schema") != SCHEMA or not manifest.get("source_revision"):
        raise ValueError("frozen corrective manifest requires schema/source revision")
    rows = manifest["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate frozen row identities")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = canonical_hash(manifest)
    stop = len(rows) if stop is None else min(stop, len(rows))
    for row in rows[start:stop]:
        directory = output / row["id"]
        directory.mkdir(parents=True, exist_ok=True)
        identity = canonical_hash((manifest_hash, row))
        status_path = directory / "completion.json"
        if status_path.exists():
            saved = json.loads(status_path.read_text())
            if saved["identity"] != identity:
                raise ValueError("incompatible corrective row resume")
            continue
        if deadline_epoch is not None and time.time() + cleanup_reserve >= deadline_epoch:
            break
        settings = {**row.get("settings", row), "source_revision": manifest["source_revision"]}
        seconds = float(settings["seconds"])
        if deadline_epoch is not None:
            seconds = min(seconds, max(0.001, deadline_epoch - time.time() - cleanup_reserve))
        settings["seconds"] = seconds
        started_path = directory / "started.json"
        if started_path.exists():
            previous = json.loads(started_path.read_text())
            if previous["identity"] != identity:
                raise ValueError("incompatible interrupted row resume")
            # An interrupted scientific row keeps its unknown outcome and full
            # recorded reservation; a replacement requires a new explicit row.
            receipt = dict(
                identity=identity,
                status="interrupted_unknown",
                reserved_seconds=previous["seconds"],
                result=None,
                cleanup_complete=None,
            )
        else:
            write_artifact(
                started_path, dict(identity=identity, seconds=seconds, started_epoch=time.time())
            )
            outcome = bounded_call(
                evaluate_row,
                row["case"],
                settings,
                str(directory),
                timeout=seconds,
                memory_mb=float(settings["memory_mb"]),
                cpu_seconds=float(settings["cpu_seconds"]),
            )
            receipt = dict(
                identity=identity,
                status=outcome.status,
                detail=outcome.detail,
                result=outcome.value,
                cleanup_complete=outcome.cleanup_complete,
                resources=dict(outcome.resource_usage),
                reserved_seconds=seconds,
            )
        write_artifact(status_path, receipt)
        if receipt["cleanup_complete"] is False:
            break
    statuses = []
    for row in rows:
        path = output / row["id"] / "completion.json"
        statuses.append(
            dict(
                id=row["id"],
                **(
                    json.loads(path.read_text())
                    if path.exists()
                    else dict(status="not_attempted", result=None)
                ),
            )
        )
    report = dict(
        schema=SCHEMA,
        manifest_hash=manifest_hash,
        expected_rows=len(rows),
        rows=statuses,
        completed_rows=sum(r["status"] != "not_attempted" for r in statuses),
        denominator_preserved=True,
        source_revision=manifest["source_revision"],
    )
    write_artifact(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    run(args.manifest, args.output, args.start, args.stop, args.deadline_epoch)


if __name__ == "__main__":
    main()
