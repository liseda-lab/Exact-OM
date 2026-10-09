"""Frozen-model value/selection diagnostics on the original supplied inventory.

No generation, model fitting, test-driven selection or external-optimum claim.
The primary generated-pool experiment and its receipts remain independent.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, make_objective, read_record
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair.expanded_profile import checkpoint
from tools.repair import fresh_evaluation as fresh

SCHEMA = "exact-repair/fresh-common-inventory/v1"


def make_rows(primary):
    rows = []
    for original in primary["rows"]:
        row = {**original, "primary_row_id": original["id"], "kind": "common_inventory_diagnostic"}
        row.pop("id")
        row["id"] = canonical_hash(row)
        rows.append(row)
    return rows


def prepare(primary_path, audit_path, output):
    primary, audit = read(primary_path), read(audit_path)
    if primary["schema"] != fresh.SCHEMA or audit["status"] != "complete":
        raise ValueError("Completed primary receipt audit required")
    schedule = dict(
        schema=SCHEMA,
        primary_schedule=binding(primary_path),
        primary_audit=binding(audit_path),
        cases=primary["cases"],
        arms=primary["arms"],
        rows=make_rows(primary),
        program=primary["program"],
        authorization=primary["authorization"],
        corpus_completion=primary["corpus_completion"],
        split_schedule=primary["split_schedule"],
        seed=primary["seed"],
        scheduled_rows=576,
        learned_rows=384,
        control_rows=192,
        parents=32,
        warm_start=False,
        fitting=False,
        generation=False,
        test_feedback_for_selection=False,
        followup="xr21-expanded-evaluation-001",
        inventory="Original observable supplied candidate IDs/axioms, no augmentation; deletion is an explicit keep/delete language ablation",
        preparation="Original effective graph, retrieval and pair limits; frozen checkpoint schema admission; no new parameters",
        scientific_budget="300 wall seconds, 600 CPU seconds, 8192 MiB per row; scoring within the existing 60-second preparation allowance",
        selection="Original native solver, verifier, shortlist and frozen plan-risk head when enabled",
        semantic_labels="Selected verified assignment only, evaluator opened after selection; original desired-query basis remains unqualified",
        value_diagnostic="Frozen unary/pair coefficients and selected quantized utility minus independently measured selected utility; selected-only error is not whole-inventory calibration",
        regret="Unavailable without complete external teacher; never use learned optimum or partial labels as semantic regret",
        control_limitation=primary["symbolic_limitation"],
        claims="Exploratory common-inventory diagnostics only; no primary generated-pool, strongest-symbolic, G0-G2 or learning-efficiency claim",
        operational_policy="16 rows per 5400-second slice, based on existing development profiling and 16*300 seconds plus receipt/setup overhead; immutable row checkpoints; no implicit replay",
        profile_timeouts_retained=primary["profile_timeouts_retained"],
        api_spend_usd=0,
    )
    validate_schedule(schedule)
    immutable(output, schedule)
    return schedule


def validate_schedule(schedule):
    primary = bound(schedule["primary_schedule"])
    audit = bound(schedule["primary_audit"])
    if (
        audit["status"] != "complete"
        or audit["scheduled_rows"] != 576
        or bound(audit["manifest"])["schedule"] != schedule["primary_schedule"]
    ):
        raise ValueError("Primary audit belongs to another schedule")
    if (
        schedule["schema"] != SCHEMA
        or len(primary["cases"]) != 64
        or len(primary["arms"]) != 9
        or len(primary["rows"]) != 576
        or sum(a["kind"] == "learned" for a in primary["arms"]) != 6
        or schedule["cases"] != primary["cases"]
        or schedule["arms"] != primary["arms"]
        or schedule["rows"] != make_rows(primary)
        or any(
            schedule[k]
            for k in ("fitting", "generation", "warm_start", "test_feedback_for_selection")
        )
    ):
        raise ValueError("Common inventory schedule or frozen selection changed")
    for key in ("program", "authorization", "corpus_completion", "split_schedule"):
        if schedule[key] != primary[key]:
            raise ValueError("Corpus or authorization identity changed")
        bound(schedule[key])
    for arm in schedule["arms"]:
        for key in ("model", "protocol", "completion", "training_report"):
            if key in arm:
                fresh.pilot.check_binding(arm[key])
        resources = bound(arm["protocol"])["resources"]
        if any(
            resources[k] != v
            for k, v in dict(
                case_wall_seconds=300, case_cpu_seconds=600, case_rss_mb=8192, generation_seconds=60
            ).items()
        ):
            raise ValueError("Original scientific budgets differ")
    for item in schedule["cases"]:
        if item["status"] == "materialized":
            problem = read_record(bound(item["observable"]))
            if problem.content_hash != item["input_hash"]:
                raise ValueError("Common inventory input changed")


def observable_graph(problem, protocol, directory):
    from exact.repair.graph import EffectivePreparation
    from exact.repair.retrieval import retrieve_vocabulary

    options = fresh.pilot.generation_options(protocol, Path(directory) / "unused-compiler-cache")
    keys = (
        "max_graph_nodes",
        "max_graph_edges",
        "max_explanations",
        "max_text_tokens",
        "pair_factor_limit_per_object",
        "pair_max_pairs",
        "pair_max_factors",
        "retrieval_config",
    )
    preparation = EffectivePreparation(**{k: options[k] for k in keys}, revision="v3")
    retrieved = retrieve_vocabulary(problem, config=preparation.retrieval_config)
    # Retrieval provides observable evidence/symbol context, but never adds candidates.
    graph = preparation.graph(problem, retrieved)
    return preparation, graph


def freeze_inventory(problem, arm, protocol, directory):
    from exact.repair.pipeline import FrozenNeuralRound, FrozenPlanRisk, model_digest

    original_inventory = canonical_hash(problem.objects)
    if arm["kind"] == "control":
        problem, objective = fresh.control_objective(problem, arm["id"], protocol)
        frozen = FrozenNeuralRound(
            problem, objective, canonical_hash(("no_learned_graph", original_inventory)), "", ()
        )
        coverage = dict(status="control", kind=arm["id"])
    else:
        import torch
        from exact.repair.model import RepairModel

        torch.set_num_threads(1)
        torch.manual_seed(13)
        path = fresh.pilot.check_binding(arm["model"])
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("model_schema") != "exact-repair/model/v3":
            raise ValueError("Native v3 frozen model required")
        model = RepairModel(state["metadata"], **state["config"])
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval()
        digest = model_digest(model)
        if digest != arm["model_hash"]:
            raise ValueError("Frozen model identity changed")
        preparation, graph = observable_graph(problem, protocol, directory)
        pairs = preparation.pairs(problem, graph, enabled=model.pair_head is not None)
        with torch.no_grad():
            memory = model.encode(graph)
            unary, factors = model.score_inventory(
                problem.objects, memory, interaction_pairs=pairs.pairs
            )
        objective = make_objective(
            problem.objects,
            tuple(tuple(float(v) for v in row) for row in unary),
            profile=tuple(sorted(protocol["objective"]["edit_weights"].items())),
            pairs=tuple((*key, float(v)) for key, v in sorted(factors.items())),
            scale=protocol["objective"]["integer_scale"],
        )
        objective = dataclasses.replace(
            objective,
            model_hash=digest,
            pair_selection_hash=canonical_hash(pairs),
            target_basis="symbolic-semantic-vector/v3",
        )
        graph_hash = canonical_hash(graph)
        risk = (
            FrozenPlanRisk(
                str(path),
                arm["model"]["sha256"],
                path.stat().st_size,
                graph,
                problem.objects,
                graph.admitted_supports,
                digest,
                graph_hash,
                original_inventory,
            )
            if model.plan_risk_enabled
            else None
        )
        frozen = FrozenNeuralRound(problem, objective, graph_hash, digest, (), risk)
        coverage = dict(
            status="scored",
            graph_hash=graph_hash,
            graph_nodes=len(graph.nodes),
            graph_edges=len(graph.edges),
            omitted_nodes=len(graph.omitted_nodes),
            omitted_supports=len(graph.omitted_supports),
            omitted_evidence=list(graph.omitted_evidence),
            support_omissions=graph.support_omissions,
            pairs=pairs.pairs,
            pair_omissions=pairs.omissions,
            pair_factors=len(factors),
            preparation_identity=preparation.content_hash,
            risk_enabled=risk is not None,
            model_hash=digest,
        )
    if arm["id"] != "deletion" and canonical_hash(frozen.problem.objects) != original_inventory:
        raise ValueError("Fixed inventory changed during scoring")
    write_artifact(
        Path(directory) / "inventory-scoring.json",
        dict(
            original_inventory_hash=original_inventory,
            scored_inventory_hash=canonical_hash(frozen.problem.objects),
            generation=False,
            fitting=False,
            evaluator_opened=False,
            coverage=coverage,
        ),
    )
    return frozen


def evaluate_row(schedule, row, directory):
    return fresh.evaluate_row(schedule, row, directory, inventory_only=True)


def run(schedule_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    schedule = read(schedule_path)
    validate_schedule(schedule)
    if not 0 <= start < stop <= len(schedule["rows"]):
        raise ValueError("Invalid common-inventory slice")
    runtime = runtime_manifest()
    identity = canonical_hash((sha(schedule_path), sha(__file__), sha(fresh.__file__), runtime))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, statuses = [], Counter()
        for row in schedule["rows"][start:stop]:
            write_artifact(output / "progress.json", dict(recorded=len(rows), row=row["id"]))
            saved = fresh.one_row(schedule, row, output, identity, evaluator=evaluate_row)
            fresh.raise_on_software_failure(saved)
            status = bound(saved["result"])["status"] if saved.get("result") else saved["status"]
            if status == "scoring_error":
                raise RuntimeError(
                    "Inventory scoring software failure: " + bound(saved["result"])["detail"]
                )
            statuses[status] += 1
            rows.append(
                dict(
                    **binding(output / "rows" / (row["id"] + ".json")),
                    row_id=row["id"],
                    status=status,
                )
            )
        return checkpoint(
            output / "report.json",
            identity,
            schema=SCHEMA,
            status="complete",
            schedule=binding(schedule_path),
            slice=[start, stop],
            scheduled=stop - start,
            recorded=len(rows),
            rows=rows,
            outcomes=dict(statuses),
            runtime=runtime,
            study_complete=False,
            gates_passed=False,
            followup=schedule["followup"],
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
