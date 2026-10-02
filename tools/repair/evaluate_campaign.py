"""Freeze and run a finite native v3 held-out evaluation, with every row retained.

The supervisor/dispatcher and immutable model batches are never modified here.
Test semantics are queried only by the worker after the schedule has been frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.checkpointing import CumulativeBudget
from exact.repair.learning import SemanticTargetSpec
from exact.repair.protocol import (
    RepairProtocolV3,
    load_protocol_v3,
    training_projection_v3,
)
from exact.repair.records import canonical_hash, canonical_json
from exact.repair.workers import bounded_call
from tools.repair.campaign_handoff import file_hash
from tools.repair.prepare import case_from_dict, case_to_dict
from tools.repair.prepare_controls import checked_preparation

SCHEMA = "exact-repair/campaign-evaluation/v3"
METHODS = ("grammar_circuit", "semantic_circuit", "semantic_enumeration_decoder")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def check_binding(value):
    if file_hash(value["path"]) != value["sha256"]:
        raise ValueError("Evaluation dependency changed: " + value["path"])
    return Path(value["path"])


def selected_model(arm, run):
    """Require a completed, development-selected v3 checkpoint; never select anew."""
    import torch

    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest

    torch.set_num_threads(1)

    completion = json.loads(Path(run["completion_path"]).read_text())
    if completion["status"] != "complete" or completion["exit_code"] != 0:
        raise ValueError("Model process is not complete")
    if (
        completion.get("step_id") != run["step_id"]
        or completion.get("dispatch_nonce") != run["dispatch_nonce"]
    ):
        raise ValueError("Model completion does not match registered launch")
    path = Path(arm["model"])
    report_path = path.with_name("report.json")
    report = json.loads(report_path.read_text())
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("model_schema") != "exact-repair/model/v3"
        or state["config"].get("revision") != "v3"
        or report["schema"] != "exact-repair/training/v3"
        or report["status"] != "complete"
        or report["checkpoint_criterion"] != "generated_pool_verified_quality_effort"
        or report["selection_fallback"] != "stop"
        or report["selected_epoch"] not in [h["epoch"] for h in report["history"]]
    ):
        raise ValueError("Require an actual development-selected native v3 model")
    chosen = next(h for h in report["history"] if h["epoch"] == report["selected_epoch"])
    if chosen["selection_status"] != "eligible":
        raise ValueError("Published checkpoint was not development-eligible")
    outputs = json.loads(Path(run["completion_path"]).with_name("outputs.json").read_text())
    if outputs["training/model.pt"] != file_hash(path) or outputs[
        "training/report.json"
    ] != file_hash(report_path):
        raise ValueError("Model artifacts differ from completed worker output receipt")
    model = RepairModel(state["metadata"], **state["config"])
    model.load_state_dict(state["state_dict"])
    if model_digest(model) != report["model_hash"]:
        raise ValueError("Selected model and training report disagree")
    return {
        "model": binding(path),
        "training_report": binding(report_path),
        "completion": binding(run["completion_path"]),
        "model_hash": report["model_hash"],
        "selected_epoch": report["selected_epoch"],
        "checkpoint_criterion": report["checkpoint_criterion"],
        "status": "available",
    }


def prepare(campaign, output):
    """Freeze metadata, schedule and dependencies without computing any test outcome."""
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve an existing evaluation schedule")
    plan = json.loads((campaign / "plan.json").read_text())
    registry = json.loads((campaign / "supervisor/registry.json").read_text())
    cases, caches, report, _, _ = checked_preparation(
        plan["labels"]["source_preparation"], plan["arms"][0]["protocol"]
    )
    tests = sorted((c for c in cases if c.split == "test"), key=lambda c: c.case_id)
    if len(tests) != plan["case_counts"]["test"] or any(c.case_id in caches for c in tests):
        raise ValueError("Reserved test split/label boundary changed")
    runs = {r["id"]: r for r in registry["runs"]}
    terminal = {r["logical_id"]: r for r in registry.get("terminal_arms", [])}
    arms = []
    for arm in plan["arms"]:
        run = runs[arm["id"] + "-001"]
        seen = set()
        while run.get("superseded_by"):
            if run["id"] in seen:
                raise ValueError("Cyclic model recovery lineage")
            seen.add(run["id"])
            run = runs[run["superseded_by"]]
        if arm["id"] in terminal:
            record = terminal[arm["id"]]
            if record["resolved_run_id"] != run["id"]:
                raise ValueError("Terminal model lineage differs")
            evidence = {
                "status": "unavailable",
                "reason": "repair_limit_exhausted",
                "accounting": binding(record["accounting_path"]),
                "completion": binding(record["completion_path"]),
            }
            if (
                evidence["accounting"]["sha256"] != record["accounting_sha256"]
                or evidence["completion"]["sha256"] != record["completion_sha256"]
            ):
                raise ValueError("Terminal accounting changed")
        else:
            evidence = selected_model(arm, run)
        arms.append(
            {"id": arm["id"], "run_id": run["id"], "protocol": binding(arm["protocol"]), **evidence}
        )
    rows = []
    for arm in arms:
        for case in tests:
            rows.append(
                dict(
                    kind="generated_pool",
                    arm_id=arm["id"],
                    case_id=case.case_id,
                    seconds=300,
                    cpu_seconds=600,
                    memory_mb=8192,
                )
            )
    for case in tests:
        for obj in case.problem.objects:
            for method in METHODS:
                rows.append(
                    dict(
                        kind="circuit",
                        method=method,
                        case_id=case.case_id,
                        object_id=obj.object_id,
                        seconds=60,
                        cpu_seconds=120,
                        memory_mb=8192,
                    )
                )
    for row in rows:
        row["id"] = canonical_hash(row)
    schedule = {
        "schema": SCHEMA,
        "campaign": str(campaign),
        "source_preparation": binding(plan["labels"]["source_preparation"]),
        "original_label_seconds": report.get("label_seconds"),
        "original_label_cpu_seconds": report.get("label_cpu_seconds"),
        "original_cache_hashes": {key: canonical_hash(value) for key, value in caches.items()},
        "plan": binding(campaign / "plan.json"),
        "arms": arms,
        "cases": [case_to_dict(case) for case in tests],
        "rows": rows,
        "seed": 13,
        "planned_model_arm_count": len(arms),
        "planned_primary_rows": len(arms) * len(tests),
        "scheduled_rows": len(rows),
        "worker_seconds": plan["budgets"]["evaluation"],
        "inner_seconds": plan["budgets"]["evaluation"] - 300,
        "cache_policy": "fresh isolated cache per scheduled row; cold/warm circuit measurements explicit",
        "decoder": "native bounded exhaustive expression decoder; uniform derivation draws; not an incremental learned decoder",
        "circuit_scope": "small matched-language support/sampling and compiler-cost diagnostic; no repair or efficiency claim",
        "primary_measure": "generated_pool_verified_quality_effort",
        "test_outcomes_opened": False,
        "test_feedback_for_selection": False,
        "claims": "exploratory smoke; no G0-G2, convergence, superiority, production transfer or XR-E completion claim",
        "external_api_cost_usd": 0,
        "production_matcher": "deferred",
    }
    write_artifact(output, schedule)
    return schedule


def generation_options(protocol, cache_directory):
    from tools.repair.train import _protocol_arguments

    projected = training_projection_v3(RepairProtocolV3.model_validate(protocol))
    arguments = _protocol_arguments(projected)
    keys = (
        "mixtures",
        "candidate_cap",
        "compile_seconds",
        "max_circuit_nodes",
        "max_depth",
        "max_constructors",
        "max_graph_nodes",
        "max_graph_edges",
        "max_explanations",
        "max_text_tokens",
        "pair_factor_limit_per_object",
        "pair_max_pairs",
        "pair_max_factors",
        "quantization_scale",
        "retrieval_config",
        "circuit_limits",
        "vtree_type",
    )
    return {
        **{key: arguments[key] for key in keys},
        "draws_per_object": arguments["development_draws_per_object"],
        "profile": tuple(sorted(protocol["objective"]["edit_weights"].items())),
        "compiler_cache_directory": str(cache_directory),
        "seed": 13,
    }


def generated_row(case, arm, protocol, directory):
    """Use native v3 preparation, exact selection and evaluator-only target queries."""
    from exact.repair.pipeline import bounded_freeze_checkpoint, repair_neural_round
    from tools.repair.train import _assignment_label

    started = time.monotonic()
    resources = protocol["resources"]
    options = generation_options(protocol, directory / "compiler-cache")
    problem = replace(
        case.problem,
        budgets=replace(
            case.problem.budgets,
            total_seconds=resources["case_wall_seconds"],
            verification_seconds=resources["verification_seconds"],
            memory_mb=resources["case_rss_mb"],
        ),
    )
    frozen = bounded_freeze_checkpoint(
        problem,
        arm["model"]["path"],
        checkpoint_sha256=arm["model"]["sha256"],
        seconds=resources["generation_seconds"],
        memory_mb=resources["case_rss_mb"],
        cpu_seconds=resources["case_cpu_seconds"],
        final_candidate_removals=dict(case.final_candidate_removals),
        **options,
    )
    row = {
        "status": "generation_" + frozen.status,
        "detail": frozen.detail,
        "logical_status": "UNKNOWN",
        "semantic_benefit": None,
        "edit_cost": None,
    }
    if frozen.status != "complete":
        return row
    value = frozen.value
    if value.model_hash != arm["model_hash"]:
        raise ValueError("Frozen inference changed the selected model")
    write_artifact(
        directory / "pool.json",
        {
            "input": value.problem.to_dict(),
            "objective": value.objective.to_dict(),
            "model_hash": value.model_hash,
            "graph_hash": value.graph_hash,
            "proposal_reports": json.loads(canonical_json(value.proposal_reports)),
        },
    )
    selection = protocol["selection"]
    remaining = max(0.001, resources["case_wall_seconds"] - (time.monotonic() - started) - 5)
    solver_seconds = max(0.001, remaining - 60)
    active = replace(
        value,
        problem=replace(
            value.problem,
            budgets=replace(
                value.problem.budgets,
                total_seconds=solver_seconds,
                max_checks=selection["max_candidate_checks"],
                max_solves=selection["max_master_solves"],
                retries=selection["retry_budget"],
            ),
        ),
    )
    if not selection["risk_ordering"]:
        active = replace(active, risk_scorer=None)
    solved = bounded_call(
        repair_neural_round,
        active,
        shortlist_size=selection["shortlist_size"],
        utility_window=selection["utility_window"],
        shortlist_seconds=selection["construction_seconds"],
        diagnose=True,
        preserve_verified_input=False,
        timeout=solver_seconds,
        memory_mb=resources["case_rss_mb"],
        cpu_seconds=resources["case_cpu_seconds"],
    )
    row.update(
        status="verification_" + solved.status,
        detail=solved.detail,
        inventory_hash=canonical_hash(value.problem.objects),
        candidate_counts=[len(obj.candidates) for obj in value.problem.objects],
    )
    if solved.status != "complete":
        return row
    result = solved.value
    write_artifact(directory / "repair.json", result.to_dict())
    row.update(
        status="evaluated",
        logical_status=result.logical_status,
        certification=result.search_status,
        checks=result.checks,
        master_solves=result.solves,
        first_verified_seconds=result.first_verified_seconds,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        stage_seconds=result.stage_seconds,
        resource_counters=result.resource_counters,
        verification_scope=result.verification_scope,
    )
    if result.assignment is not None and result.logical_status == "VERIFIED_FEASIBLE":
        if result.verification is None or not result.verification.authorizes:
            raise ValueError("Verified repair lacks authorizing evidence")
        weights = protocol["teacher"]["family_weights"]
        target = SemanticTargetSpec(
            canonical_hash(case.probes), weights["desired"], weights["unwanted"]
        )
        remaining = resources["case_wall_seconds"] - (time.monotonic() - started) - 5
        if remaining > 0:
            labeled = bounded_call(
                _assignment_label,
                replace(case, problem=value.problem),
                result.assignment,
                options["profile"],
                semantic_target=target,
                timeout=remaining,
                memory_mb=resources["case_rss_mb"],
                cpu_seconds=resources["case_cpu_seconds"],
            )
            row["semantic_status"] = labeled.status
            if labeled.status == "complete":
                label = labeled.value
                write_artifact(directory / "selected-label.json", json.loads(canonical_json(label)))
                row.update(
                    semantic_status="known" if label.usable else "unknown",
                    semantic_benefit=label.benefit,
                    edit_cost=label.cost,
                    selected_utility=label.benefit - label.cost if label.usable else None,
                    semantic_target_hash=target.content_hash,
                )
        else:
            row["semantic_status"] = "unvisited_deadline"
    return row


def circuit_row(case, row, protocol, directory):
    """Compare native finite decoders on identical menus and proved conditions."""
    import torch

    from exact.repair.candidates import materialize_retrieved_endpoints
    from exact.repair.circuit import FactoredConditionedMixture
    from exact.repair.grammar import (
        compile_families,
        mapping_grammar,
        with_immutable_context,
    )
    from exact.repair.proposals import enumerate_grammar
    from exact.repair.retrieval import retrieve_vocabulary

    started = time.monotonic()
    options = generation_options(protocol, directory / "compiler-cache")
    retrieval = retrieve_vocabulary(case.problem, config=options["retrieval_config"])
    obj = next(obj for obj in case.problem.objects if obj.object_id == row["object_id"])
    menu = next(menu for menu in retrieval.menus if menu.object_id == obj.object_id)
    # Same endpoint producer and stage-0 language used by native model inference.
    expanded = materialize_retrieved_endpoints(case.problem, retrieval)
    obj = next(obj for obj in expanded.objects if obj.object_id == row["object_id"])
    encoding = mapping_grammar(
        obj,
        menu.classes,
        menu.properties,
        max_depth=options["max_depth"],
        max_constructors=options["max_constructors"],
        source_classes=menu.source_classes,
        target_classes=menu.target_classes,
        source_properties=menu.source_properties,
        target_properties=menu.target_properties,
        constraint_identity=canonical_hash((case.problem.policy, menu)),
    )
    if row["method"] != "grammar_circuit":
        encoding = with_immutable_context(encoding, case.problem.fixed_axioms, case.problem.policy)
    output = {
        "language_hash": encoding.content_hash,
        "retrieval_hash": canonical_hash(retrieval),
        "context_proofs": [proof.content_hash for proof in encoding.context_proofs],
        "context_checks": encoding.contextual_checks,
        "context_truncated": encoding.contextual_truncated,
        "preparation_seconds": time.monotonic() - started,
        "draws": 32,
        "seed": 13,
        "candidate_cap": 64,
        "logical_status": "NOT_EVALUATED",
    }
    if row["method"] == "semantic_enumeration_decoder":
        before = time.monotonic()
        enumeration = enumerate_grammar(
            encoding, max_expressions=10000, deadline=started + row["seconds"] - 3
        )
        candidates = enumeration.candidates
        weights = [len(encoding.candidate_assignments(candidate)) for candidate in candidates]
        sampled = random.Random(13).choices(candidates, weights=weights, k=32) if candidates else []
        output.update(
            status="complete",
            support_complete=True,
            support_ids=sorted(candidate.candidate_id for candidate in candidates),
            decode_seconds=time.monotonic() - before,
            sampled_ids=[candidate.candidate_id for candidate in sampled],
        )
    else:
        limits = dict(options["circuit_limits"])
        limits["aggregate_seconds"] = max(0.001, row["seconds"] - (time.monotonic() - started) - 3)
        compiled = compile_families(
            encoding,
            max_nodes=options["max_circuit_nodes"],
            vtree_type=options["vtree_type"],
            cache_directory=str(directory / "compiler-cache"),
            circuit_limits=limits,
        )
        distribution = FactoredConditionedMixture(
            compiled, torch.zeros((4, encoding.variable_count))
        )
        samples = distribution.sample(32, seed=13)
        families = [
            {
                "name": family.name,
                "status": family.status,
                "telemetry": dict(family.circuit.telemetry) if family.circuit else None,
            }
            for family in compiled.families
        ]
        output.update(
            status=(
                "complete"
                if all(f["status"] in {"resolved", "empty_language"} for f in families)
                else "partial"
            ),
            support_complete=all(f["status"] in {"resolved", "empty_language"} for f in families),
            cold_seconds=compiled.compilation_seconds,
            families=families,
            sampled_ids=[sample.candidate_id for sample in samples],
        )
        remaining = row["seconds"] - (time.monotonic() - started) - 3
        if remaining > 0:
            limits["aggregate_seconds"] = remaining
            warm = compile_families(
                encoding,
                max_nodes=options["max_circuit_nodes"],
                vtree_type=options["vtree_type"],
                cache_directory=str(directory / "compiler-cache"),
                circuit_limits=limits,
            )
            output["warm_seconds"] = warm.compilation_seconds
            output["warm_families"] = [
                {
                    "name": f.name,
                    "status": f.status,
                    "telemetry": dict(f.circuit.telemetry) if f.circuit else None,
                }
                for f in warm.families
            ]
        else:
            output["warm_status"] = "unvisited_deadline"
    output["unique_sampled"] = len(set(output["sampled_ids"]))
    output["duplicate_draws"] = len(output["sampled_ids"]) - output["unique_sampled"]
    return output


def evaluate_row(schedule, row, directory):
    """Publish potentially large evidence locally; transport only its bound receipt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case = next(
        case_from_dict(c) for c in schedule["cases"] if c["case"]["case_id"] == row["case_id"]
    )
    arm = next((a for a in schedule["arms"] if a["id"] == row.get("arm_id")), schedule["arms"][0])
    protocol = load_protocol_v3(check_binding(arm["protocol"]), for_execution=True).model_dump()
    result = (
        generated_row(case, arm, protocol, directory)
        if row["kind"] == "generated_pool"
        else circuit_row(case, row, protocol, directory)
    )
    result.update(
        row_id=row["id"],
        schedule_hash=canonical_hash(schedule),
        case_id=case.case_id,
        parent=case.structural_parent,
        kind=row["kind"],
    )
    write_artifact(directory / "result.json", result)
    return binding(directory / "result.json")


def run(schedule_path, output):
    schedule, output = json.loads(Path(schedule_path).read_text()), Path(output)
    if schedule["schema"] != SCHEMA:
        raise ValueError("Native v3 evaluation schedule required")
    for item in [schedule["source_preparation"], schedule["plan"]]:
        check_binding(item)
    for arm in schedule["arms"]:
        for key in ("model", "protocol", "training_report", "completion", "accounting"):
            if key in arm:
                check_binding(arm[key])
    identity = canonical_hash(schedule)
    results = []
    with CumulativeBudget(
        output / "evaluation-budget.json", identity, schedule["inner_seconds"]
    ) as total:
        total.begin()
        for row in schedule["rows"]:
            directory = output / "rows" / row["id"]
            receipt = directory / "receipt.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["schedule_hash"] != identity or saved["row_id"] != row["id"]:
                    raise ValueError("Saved evaluation row dependency changed")
                if saved.get("artifact"):
                    check_binding(saved["artifact"])
                results.append(saved)
                continue
            arm = next((a for a in schedule["arms"] if a["id"] == row.get("arm_id")), None)
            saved = {
                **row,
                "row_id": row["id"],
                "schedule_hash": identity,
                "status": "unvisited",
                "logical_status": "UNKNOWN",
                "artifact": None,
            }
            if arm is not None and arm["status"] == "unavailable":
                saved.update(status="unavailable", reason=arm["reason"])
            elif total.remaining > row["seconds"] + 5:
                with CumulativeBudget(
                    directory / "budget.json",
                    canonical_hash((identity, row)),
                    row["seconds"],
                    cpu_seconds=row["cpu_seconds"],
                ) as budget:
                    if budget.remaining > 0:
                        limit = budget.begin()
                        outcome = bounded_call(
                            evaluate_row,
                            schedule,
                            row,
                            str(directory),
                            timeout=limit,
                            memory_mb=row["memory_mb"],
                            cpu_seconds=budget.remaining_cpu,
                        )
                        budget.finish(
                            outcome.status,
                            cpu_seconds=dict(outcome.resource_usage).get("cpu_seconds"),
                        )
                        saved.update(
                            status=outcome.status,
                            detail=outcome.detail,
                            resources=dict(outcome.resource_usage),
                        )
                        if outcome.status == "complete":
                            artifact = outcome.value
                            payload = json.loads(check_binding(artifact).read_text())
                            if (
                                payload["schedule_hash"] != identity
                                or payload["row_id"] != row["id"]
                            ):
                                raise ValueError("Evaluation output dependency mismatch")
                            saved.update(
                                status=payload["status"],
                                artifact=artifact,
                                logical_status=payload.get("logical_status", "NOT_EVALUATED"),
                            )
                    else:
                        saved.update(
                            status="unavailable", reason="cumulative_row_allowance_exhausted"
                        )
            write_artifact(receipt, saved)
            results.append(saved)
            write_artifact(
                output / "progress.json",
                {"scheduled": len(schedule["rows"]), "recorded": len(results)},
            )
        total.finish()
    report = {
        "schema": SCHEMA,
        "schedule": binding(schedule_path),
        "scheduled": len(schedule["rows"]),
        "recorded": len(results),
        "rows": results,
        "statuses": dict(Counter(r["status"] for r in results)),
        "planned_model_arm_count": schedule["planned_model_arm_count"],
        "planned_primary_rows": schedule["planned_primary_rows"],
        "process_status": "complete",
        "scientific_status": "exploratory_not_established",
        "production_matcher": "deferred",
        "external_api_cost_usd": 0,
    }
    write_artifact(output / "evaluation-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = (
        prepare(args.source, args.output)
        if args.command == "prepare"
        else run(args.source, args.output)
    )
    print(json.dumps({key: result[key] for key in ("schema", "planned_model_arm_count")}))


if __name__ == "__main__":
    main()
