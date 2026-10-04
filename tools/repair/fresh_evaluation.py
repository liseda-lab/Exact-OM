"""Receipt-bound fresh-case evaluation; frozen pilot weights, no fitting or split reuse.

Generation and selection see only the observable RepairInputV3. Independent
semantic queries are opened after a selected assignment has been verified.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import (
    canonical_hash,
    canonical_json,
    read_record,
    replace_inventory,
    make_objective,
)
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound, immutable, audit_release_rows, TARGETS
from tools.repair.expanded_profile import checked_checkpoint, checkpoint, parent_fingerprints
from tools.repair.historical_audit import validate_attempt
from tools.repair import evaluate_campaign as pilot
from tools.repair.prepare import case_from_dict, load_preparation

SCHEMA = "exact-repair/fresh-evaluation/v1"
CONTROLS = ("symbolic_rich_action", "uniform", "deletion")


def validate_corpus(attempt):
    """Bind actual execution and every release, preserving all missing target rows."""
    report, _ = validate_attempt(attempt)
    if (
        report["status"],
        report["scheduled_cases"],
        report["training_target_cases"],
        report["fresh_evaluation_target_cases"],
        report["teacher_queries"],
        report["test_outcomes_opened"],
    ) != ("complete", 288, 224, 64, 0, False):
        raise ValueError("Corpus completion scope differs")
    inventory = bound(report["structural_audit"])
    split = bound(report["split_schedule"])
    if canonical_hash(split["selected"]) != canonical_hash(inventory["selected"]):
        raise ValueError("Corpus structural split differs")
    all_rows, releases = [], {}
    for name, target in TARGETS.items():
        release = bound(report["releases"][name]["manifest"])
        prep = Path(report["releases"][name]["preparation"]["path"])
        bound(report["releases"][name]["preparation"])
        cases, caches, info = load_preparation(prep)
        if (release["split"], release["scheduled"], len(release["rows"])) != (name, target, target):
            raise ValueError("Corpus release denominator differs")
        if caches or canonical_hash(info) != canonical_hash(release):
            raise ValueError("Unexpected labels or release/preparation mismatch")
        materialized = [r for r in release["rows"] if r["status"] == "materialized"]
        if {c.case_id for c in cases} != {r["case_id"] for r in materialized}:
            raise ValueError("Preparation omitted materialized rows")
        by_id = {c.case_id: c for c in cases}
        for row in materialized:
            case = case_from_dict(bound(row["evaluator"]))
            observable = read_record(bound(row["observable"]))
            if (case.content_hash if hasattr(case, "content_hash") else canonical_hash(case)) != (
                by_id[case.case_id].content_hash
                if hasattr(case, "content_hash")
                else canonical_hash(by_id[case.case_id])
            ):
                raise ValueError("Preparation/evaluator case differs")
            if (
                case.problem.content_hash != row["input_hash"]
                or observable != case.problem
                or case.schema_revision != "v3"
                or case.final_candidate_removals
                or case.structural_parent != row["structural_parent"]
                or case.split != ("test" if name == "fresh_evaluation" else name)
                or set(parent_fingerprints(case)) != set(row["fingerprints"])
            ):
                raise ValueError("Native release identity or structural fingerprint differs")
        all_rows.extend(release["rows"])
        releases[name] = release
    audit_release_rows(all_rows, inventory)
    fresh = releases["fresh_evaluation"]["rows"]
    actual = [r for r in fresh if r["status"] == "materialized"]
    groups = {r["group_id"] for r in actual}
    for group in groups:
        if Counter(r["control"] for r in actual if r["group_id"] == group) != {
            "coherent": 1,
            "corrupted": 1,
        }:
            raise ValueError("Fresh parent lacks paired clean/corrupted variants")
    return report, fresh


def freeze_models(campaign, model_schedule):
    """Reuse model selections only; never adopt the old evaluation's cases or budgets."""
    registry = read(Path(campaign) / "supervisor/registry.json")
    previous = bound(model_schedule)
    arms = previous["arms"]
    if len(arms) != 6 or len({a["id"] for a in arms}) != 6:
        raise ValueError("Six frozen pilot arms required")
    runs = {r["id"]: r for r in registry["runs"]}
    selected = []
    for arm in arms:
        for key in ("model", "protocol", "completion", "training_report"):
            pilot.check_binding(arm[key])
        run = runs[arm["run_id"]]
        if run.get("superseded_by"):
            raise ValueError(
                "Selected pilot lineage changed; explicit compatibility review required"
            )
        evidence = pilot.selected_model(dict(model=arm["model"]["path"]), run)
        if any(evidence[k] != arm[k] for k in evidence):
            raise ValueError("Pilot checkpoint selection changed")
        selected.append(dict(arm, kind="learned"))
    return selected


def make_rows(cases, arms):
    rows = []
    for arm in arms:
        for ordinal, case in enumerate(cases):
            row = dict(
                arm_id=arm["id"],
                case_index=ordinal,
                case_id=case.get("case_id"),
                seconds=300,
                cpu_seconds=600,
                memory_mb=8192,
            )
            row["id"] = canonical_hash(row)
            rows.append(row)
    return rows


def prepare(campaign, model_schedule, corpus_attempt, output):
    report, cases = validate_corpus(corpus_attempt)
    arms = freeze_models(campaign, model_schedule)
    # Only inference fields are imported. Old identity/input/split/campaign caps
    # never define the fresh study. Check matching declared scientific resources.
    common = None
    for arm in arms:
        protocol = bound(arm["protocol"])
        fixed = {
            k: protocol[k] for k in ("resources", "generation", "selection", "objective", "teacher")
        }
        if common is None:
            common = fixed
        elif fixed != common:
            raise ValueError("Pilot inference scientific settings are not matched")
        if any(
            protocol["resources"][k] != v
            for k, v in dict(
                case_wall_seconds=300, case_cpu_seconds=600, case_rss_mb=8192, generation_seconds=60
            ).items()
        ):
            raise ValueError("Original per-case comparison allowance changed")
    control_protocol = arms[-1]["protocol"]
    arms.extend(
        dict(id=name, kind="control", protocol=control_protocol, status="available")
        for name in CONTROLS
    )
    registry = read(Path(campaign) / "supervisor/registry.json")
    schedule = dict(
        schema=SCHEMA,
        program=registry["expanded_program"],
        authorization=registry["time_limit_amendment"],
        corpus_attempt=corpus_attempt,
        corpus_completion=corpus_attempt["report"],
        releases=report["releases"],
        split_schedule=report["split_schedule"],
        model_selection_source=model_schedule,
        cases=cases,
        arms=arms,
        rows=make_rows(cases, arms),
        seed=13,
        planned_cases=64,
        planned_model_rows=384,
        planned_control_rows=192,
        scheduled_rows=576,
        primary_measure="generated_pool_verified_quality_effort",
        test_outcomes_opened=False,
        test_feedback_for_selection=False,
        warm_start=False,
        api_spend_usd=0,
        inference_projection="Only original per-case generation/objective/selection/teacher settings; pilot training/test identities and total budgets do not apply",
        control_definition=dict(
            symbolic_rich_action="Uniform grammar proposals, original-axiom retention fraction minus declared edit cost; native semantic constraints, verifier and cuts; historical heuristic, not strongest symbolic semantic baseline",
            uniform="Uniform grammar proposals and negative declared edit cost",
            deletion="Same charged generated pool followed by keep/delete-only filter; negative declared edit cost; language ablation",
        ),
        control_generator="Zero-parameter disposable no-graph scaffold; uniform proposals ignore weights, readouts discarded, no fitting or pilot checkpoint",
        symbolic_limitation="No superiority claim against a fully developed symbolic semantics baseline; final audit must retain this limitation and any missing required control",
        evaluation_labels="Selected-plan native typed consequence vector only, after independent verification; no training labels or evaluator features",
        common_inventory="Separate follow-up diagnostic, not a substitute for primary generated pools; final stage retained",
        cache_policy="Isolated cold cache for every case/arm; no cross-arm reuse",
        operational_policy="16 rows per 5400-second worker slice (16*300 plus setup/cleanup); resume immutable receipts only; inflight rows require explicit ownership and budget reconciliation",
        gate_status="G0-G2 not established; exploratory fresh frozen-model transfer only; no learning-efficiency claim",
        coverage="Eight declared constructions and path variants only; use original seen/unseen family metadata and parent grouping",
        profile_timeouts_retained=4,
    )
    immutable(output, schedule)
    return schedule


def control_objective(problem, arm, protocol):
    """No evaluator labels, hidden clean theory or learned coefficients enter controls."""
    if arm == "deletion":
        objects = []
        for obj in problem.objects:
            candidates = tuple(
                c
                for c in obj.candidates
                if (
                    "keep" in c.action_tags
                    and set(c.axioms) == set(obj.original_axioms)
                    and not c.active_expressions
                )
                or (
                    obj.eligible
                    and not obj.locked
                    and not c.axioms
                    and not c.active_expressions
                    and "delete" in c.action_tags
                )
            )
            if not candidates:
                raise ValueError("Deletion control lacks an unchanged original state")
            objects.append(dataclasses.replace(obj, candidates=candidates))
        problem = replace_inventory(problem, tuple(objects))
    values = tuple(
        tuple(
            (
                len(set(c.axioms) & set(o.original_axioms)) / max(1, len(set(o.original_axioms)))
                if arm == "symbolic_rich_action"
                else 0.0
            )
            for c in o.candidates
        )
        for o in problem.objects
    )
    objective = make_objective(
        problem.objects,
        values,
        profile=tuple(sorted(protocol["objective"]["edit_weights"].items())),
        scale=protocol["objective"]["integer_scale"],
    )
    return problem, objective


def generate_control(problem, arm, protocol, directory, generation_overrides=None):
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.retrieval import retrieve_vocabulary

    torch.set_num_threads(1)
    torch.manual_seed(13)
    options = pilot.generation_options(protocol, Path(directory) / "compiler-cache")
    options.update(generation_overrides or {})
    preparation = EffectivePreparation(
        max_graph_nodes=options["max_graph_nodes"],
        max_graph_edges=options["max_graph_edges"],
        max_explanations=options["max_explanations"],
        max_text_tokens=options["max_text_tokens"],
        pair_factor_limit_per_object=options["pair_factor_limit_per_object"],
        pair_max_pairs=options["pair_max_pairs"],
        pair_max_factors=options["pair_max_factors"],
        retrieval_config=options["retrieval_config"],
        revision="v3",
    )
    graph = preparation.graph(
        problem, retrieve_vocabulary(problem, config=options["retrieval_config"])
    )
    model = RepairModel(
        graph.metadata,
        encoder="none",
        hidden_dim=8,
        layers=0,
        heads=2,
        dropout=0,
        pairwise=False,
        plan_risk=False,
        revision="v3",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    frozen = freeze_neural_round(
        problem,
        model,
        graph=None if generation_overrides else graph,
        proposal_arm="grammar_uniform",
        **options,
    )
    problem, objective = control_objective(frozen.problem, arm, protocol)
    return dataclasses.replace(frozen, problem=problem, objective=objective, risk_scorer=None)


def ensure_cleanup(outcome):
    if not outcome.cleanup_complete:
        raise RuntimeError(
            "Nested worker cleanup incomplete; reconcile descendants before continuation"
        )


def evaluate_row(schedule, row, directory, *, remaining_budget=None, inventory_only=False):
    """Full comparison call including generation, solving and evaluator-only queries."""
    from exact.repair.pipeline import bounded_freeze_checkpoint, repair_neural_round
    from exact.repair.workers import bounded_call
    from exact.repair.learning import SemanticTargetSpec
    from exact.repair.protocol import load_protocol_v3
    from tools.repair.train import _assignment_label

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    item = schedule["cases"][row["case_index"]]
    arm = next(a for a in schedule["arms"] if a["id"] == row["arm_id"])
    protocol = load_protocol_v3(Path(arm["protocol"]["path"]), for_execution=True).model_dump(
        by_alias=True
    )
    resources, selection = dict(protocol["resources"]), protocol["selection"]
    if remaining_budget is not None:
        from tools.repair.schema_recovery import validate_remaining_budget

        validate_remaining_budget(remaining_budget, row, resources)
        resources["case_wall_seconds"] = remaining_budget["wall_seconds"]
        resources["generation_seconds"] = remaining_budget["generation_seconds"]
    # The evaluator record is deliberately unopened until selection completes.
    problem = read_record(bound(item["observable"]))
    from tools.repair.robustness import generation_overrides, audit_final_pool

    overrides = generation_overrides(item, problem)
    problem = dataclasses.replace(
        problem,
        budgets=dataclasses.replace(
            problem.budgets,
            total_seconds=resources["case_wall_seconds"],
            verification_seconds=resources["verification_seconds"],
            memory_mb=resources["case_rss_mb"],
        ),
    )
    if inventory_only:
        from tools.repair.common_inventory import freeze_inventory

        generated = bounded_call(
            freeze_inventory, problem, arm, protocol, str(directory),
            timeout=resources["generation_seconds"],
            memory_mb=row["memory_mb"], cpu_seconds=row["cpu_seconds"],
        )
    elif arm["kind"] == "learned":
        generated = bounded_freeze_checkpoint(
            problem,
            arm["model"]["path"],
            checkpoint_sha256=arm["model"]["sha256"],
            seconds=resources["generation_seconds"],
            memory_mb=row["memory_mb"],
            cpu_seconds=row["cpu_seconds"],
            **{**pilot.generation_options(protocol, directory / "compiler-cache"), **overrides},
        )
    else:
        generated = bounded_call(
            generate_control,
            problem,
            arm["id"],
            protocol,
            str(directory),
            generation_overrides=overrides or None,
            timeout=resources["generation_seconds"],
            memory_mb=row["memory_mb"],
            cpu_seconds=row["cpu_seconds"],
        )
    ensure_cleanup(generated)
    result = dict(
        status="generation_" + generated.status,
        detail=generated.detail,
        logical_status="UNKNOWN",
        semantic_benefit=None,
        edit_cost=None,
        semantic_status="not_evaluated",
        generation_resources=dict(generated.resource_usage),
    )
    if generated.status == "error" and generated.detail == "ValueError: cannot infer complete original relation for endpoint retrieval":
        result["status"] = "unsupported_original_mapping_bundle"
    if generated.status == "error" and generated.detail.startswith("ModelGraphSchemaError:"):
        result["status"] = "unavailable_model_schema"
        result["compatibility_scope"] = "Frozen graph encoder lacks required relation or node parameters; no weights changed"
    if generated.status == "complete":
        frozen = generated.value
        result["intervention_audit"] = audit_final_pool(frozen.problem, item)
        if arm["kind"] == "learned" and frozen.model_hash != arm["model_hash"]:
            raise ValueError("Frozen model identity changed during generation")
        write_artifact(
            directory / "pool.json",
            dict(
                input=frozen.problem.to_dict(),
                objective=frozen.objective.to_dict(),
                model_hash=frozen.model_hash,
                graph_hash=frozen.graph_hash,
                proposal_reports=json.loads(canonical_json(frozen.proposal_reports)),
            ),
        )
        result.update(
            inventory_hash=canonical_hash(frozen.problem.objects),
            candidate_counts=[len(o.candidates) for o in frozen.problem.objects],
            generation_statuses=[
                r.get("generation_status", "UNKNOWN") for r in frozen.proposal_reports
            ],
            distribution_scopes=[
                r.get("distribution_scope", "unknown") for r in frozen.proposal_reports
            ],
        )
        remaining = resources["case_wall_seconds"] - (time.monotonic() - started) - 5
        solver_seconds = max(0.001, remaining - 60)
        active = dataclasses.replace(
            frozen,
            problem=dataclasses.replace(
                frozen.problem,
                budgets=dataclasses.replace(
                    frozen.problem.budgets,
                    total_seconds=solver_seconds,
                    max_checks=selection["max_candidate_checks"],
                    max_solves=selection["max_master_solves"],
                    retries=selection["retry_budget"],
                ),
            ),
        )
        solved = bounded_call(
            repair_neural_round,
            active,
            shortlist_size=selection["shortlist_size"],
            utility_window=selection["utility_window"],
            shortlist_seconds=selection["construction_seconds"],
            diagnose=True,
            preserve_verified_input=False,
            ledger_path=directory / "search-ledger.json",
            timeout=solver_seconds,
            memory_mb=row["memory_mb"],
            cpu_seconds=row["cpu_seconds"],
        )
        ensure_cleanup(solved)
        result.update(
            status="verification_" + solved.status,
            detail=solved.detail,
            selection_resources=dict(solved.resource_usage),
        )
        if solved.status == "complete":
            repair = solved.value
            write_artifact(directory / "repair.json", repair.to_dict())
            result.update(
                status="evaluated",
                logical_status=repair.logical_status,
                certification=repair.search_status,
                checks=repair.checks,
                master_solves=repair.solves,
                first_verified_seconds=repair.first_verified_seconds,
                lower_bound=repair.lower_bound,
                upper_bound=repair.upper_bound,
                stage_seconds=repair.stage_seconds,
                resource_counters=repair.resource_counters,
                verification_scope=repair.verification_scope,
            )
            if repair.assignment is not None and repair.logical_status == "VERIFIED_FEASIBLE":
                if repair.verification is None or not repair.verification.authorizes:
                    raise ValueError("Missing authorizing verification")
                if inventory_only:
                    result["selected_objective_utility"] = (
                        frozen.objective.score(repair.assignment) / frozen.objective.scale
                    )
                    result["selected_assignment"] = list(repair.assignment)
                case = case_from_dict(bound(item["evaluator"]))
                weights = protocol["teacher"]["family_weights"]
                target = SemanticTargetSpec(
                    canonical_hash(case.probes), weights["desired"], weights["unwanted"]
                )
                remaining = resources["case_wall_seconds"] - (time.monotonic() - started) - 5
                if remaining > 0:
                    labeled = bounded_call(
                        _assignment_label,
                        dataclasses.replace(case, problem=frozen.problem),
                        repair.assignment,
                        tuple(sorted(protocol["objective"]["edit_weights"].items())),
                        semantic_target=target,
                        timeout=remaining,
                        memory_mb=row["memory_mb"],
                        cpu_seconds=row["cpu_seconds"],
                    )
                    ensure_cleanup(labeled)
                    result.update(
                        semantic_status=labeled.status,
                        semantic_detail=labeled.detail,
                        evaluator_resources=dict(labeled.resource_usage),
                    )
                    if labeled.status == "complete":
                        label = labeled.value
                        write_artifact(
                            directory / "selected-label.json", json.loads(canonical_json(label))
                        )
                        result.update(
                            semantic_status="known" if label.usable else "unknown",
                            semantic_benefit=label.benefit,
                            edit_cost=label.cost,
                            selected_utility=label.benefit - label.cost if label.usable else None,
                            semantic_target_hash=target.content_hash,
                        )
                else:
                    result["semantic_status"] = "unvisited_deadline"
    result.update(
        row_id=row["id"],
        schedule_hash=canonical_hash(schedule),
        case_id=item["case_id"],
        base_case_id=item.get("base_case_id", item["case_id"]),
        condition=item.get("condition", "baseline"),
        group_id=item.get("group_id", item["structural_parent"]),
        parent=item["structural_parent"],
        family=item["family"],
        family_exposure=item["family_exposure"],
        elapsed_seconds=time.monotonic() - started,
    )
    if remaining_budget is not None:
        result["resource_recovery"] = dict(remaining_budget)
    if inventory_only:
        result["inventory_scoring_resources"] = result.pop("generation_resources")
        if result["status"].startswith("generation_"):
            result["status"] = result["status"].replace("generation_", "scoring_", 1)
        result.update(
            study_kind="common_inventory_diagnostic",
            common_inventory_regret=None,
            regret_status="unavailable_no_complete_external_teacher",
            value_error_scope="selected verified assignment only; semantic query scope unqualified",
            selected_utility_error=(
                result["selected_objective_utility"] - result["selected_utility"]
                if arm["kind"] == "learned" and result.get("selected_utility") is not None
                else None
            ),
        )
    write_artifact(directory / "result.json", result)
    return binding(directory / "result.json")


def validate_payloads(saved):
    for item in saved.get("payloads", []):
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Completed evaluation payload changed")
    if saved.get("result"):
        result = bound(saved["result"])
        if result["row_id"] != saved["row"]["id"]:
            raise ValueError("Evaluation payload row identity differs")


def one_row(schedule, row, output, identity, *, evaluator=None):
    from exact.repair.workers import bounded_call

    key = row["id"]
    path = output / "rows" / (key + ".json")
    row_identity = canonical_hash((identity, row))
    saved = checked_checkpoint(path, row_identity)
    if saved is None:
        guard = output / "inflight" / (key + ".json")
        if guard.exists():
            raise RuntimeError(
                "Unfinished fixed-budget row requires owner/ledger/cleanup reconciliation"
            )
        item = schedule["cases"][row["case_index"]]
        arm = next(a for a in schedule["arms"] if a["id"] == row["arm_id"])
        common = dict(
            row=row,
            cleanup_complete=True,
            elapsed_seconds=0,
            resources={},
            result=None,
            payloads=[],
        )
        if item["status"] != "materialized" or arm["status"] != "available":
            saved = checkpoint(
                path,
                row_identity,
                **common,
                status="unavailable",
                reason=item["status"] if item["status"] != "materialized" else arm["status"],
            )
        else:
            payload = output / "payloads" / key
            checkpoint(guard, row_identity, row=row, started_epoch=time.time())
            start = time.monotonic()
            outcome = bounded_call(
                evaluator or evaluate_row,
                schedule,
                row,
                str(payload),
                timeout=row["seconds"],
                cpu_seconds=row["cpu_seconds"],
                memory_mb=row["memory_mb"],
            )
            common.update(
                cleanup_complete=outcome.cleanup_complete,
                elapsed_seconds=time.monotonic() - start,
                resources=dict(outcome.resource_usage),
                result=outcome.value if outcome.status == "complete" else None,
                payloads=[
                    binding(p)
                    for p in sorted(payload.rglob("*"))
                    if p.is_file() and p.suffix != ".lock"
                ],
            )
            saved = checkpoint(
                path, row_identity, **common, status=outcome.status, detail=outcome.detail
            )
            validate_payloads(saved)
            # Nested cleanup failures are not made retryable by a clean outer process.
            if not outcome.cleanup_complete or "cleanup incomplete" in outcome.detail:
                raise RuntimeError("Worker cleanup incomplete; retain inflight ownership guard")
            guard.unlink()
    validate_payloads(saved)
    if not saved["cleanup_complete"]:
        raise RuntimeError("Prior cleanup incomplete")
    if (output / "inflight" / (key + ".json")).exists():
        raise RuntimeError("Saved row still requires inflight reconciliation")
    return saved


def raise_on_software_failure(saved):
    """Stop after persisting a failed row; expected scientific limits are outcomes."""
    from exact.experiments.science_health import software_failure

    candidates = [(saved.get("status"), saved.get("detail", ""))]
    if saved.get("result"):
        payload = bound(saved["result"])
        candidates.extend([(payload.get("status"), payload.get("detail", "")),
                           (payload.get("semantic_status"), payload.get("semantic_detail", ""))])
    for status, detail in candidates:
        if software_failure(status, detail):
            raise RuntimeError("Scientific worker software failure: " + (detail or str(status)))


def run(schedule_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    schedule = read(schedule_path)
    output = Path(output)
    if schedule["schema"] != SCHEMA or not 0 <= start < stop <= len(schedule["rows"]):
        raise ValueError("Invalid fresh evaluation shard")
    for ref in (
        schedule["program"],
        schedule["authorization"],
        schedule["corpus_completion"],
        schedule["split_schedule"],
    ):
        bound(ref)
    for arm in schedule["arms"]:
        for key in ("model", "protocol", "completion", "training_report"):
            if key in arm:
                pilot.check_binding(arm[key])
    runtime = runtime_manifest()
    identity = canonical_hash((sha(schedule_path), sha(__file__), runtime))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for row in schedule["rows"][start:stop]:
            write_artifact(
                output / "progress.json",
                dict(stage="evaluation", slice=[start, stop], recorded=len(rows), row=row["id"]),
            )
            saved = one_row(schedule, row, output, identity)
            raise_on_software_failure(saved)
            rows.append(
                dict(
                    **binding(output / "rows" / (row["id"] + ".json")),
                    row_id=row["id"],
                    status=saved["status"],
                )
            )
        return checkpoint(
            output / "report.json",
            identity,
            schema=SCHEMA,
            schedule=binding(schedule_path),
            status="complete",
            slice=[start, stop],
            scheduled=stop - start,
            recorded=len(rows),
            rows=rows,
            outcomes=dict(Counter(r["status"] for r in rows)),
            runtime=runtime,
            study_complete=False,
            followup=schedule.get("followup", "xr21-expanded-evaluation-001"),
            gates_passed=False,
            api_spend_usd=0,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    run(args.schedule, args.output, args.start, args.stop)


if __name__ == "__main__":
    main()
