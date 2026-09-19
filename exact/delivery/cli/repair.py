"""Opt-in standalone repair and repair of existing matching run outputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Sequence


def _prepare_input(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load and construct records inside the supervised preparation worker."""
    from exact.repair.api import matching_run_evidence, prepare_repair
    from exact.repair.records import (
        BudgetsV2,
        ObjectiveV2,
        RepairInputV2,
        make_objective,
        read_record,
    )

    if args.problem:
        payload = json.loads(args.problem.read_text())
        problem, objective = read_record(payload["input"]), read_record(payload["objective"])
        if not isinstance(problem, RepairInputV2) or not isinstance(objective, ObjectiveV2):
            raise ValueError("--problem requires v2 input and objective records")
        return problem, objective
    import pyowl_core as owl

    if args.matching_run:
        mappings, evidence = matching_run_evidence(args.matching_run)
    else:
        from exact.utils.data import read_table

        if args.alignment.suffix == ".json":
            mappings = json.loads(args.alignment.read_text())
        elif args.alignment.suffix.lower() in {".rdf", ".xml"}:
            from exact.io.writers.oaei_rdf import read_alignment

            mappings = read_alignment(args.alignment)
        else:
            mappings = read_table(args.alignment)
        evidence = {}
    if args.evidence:
        evidence.update(json.loads(args.evidence.read_text()))
    problem = prepare_repair(
        owl.load_snapshot(str(args.source)),
        owl.load_snapshot(str(args.target)),
        mappings,
        evidence=evidence,
        matcher_identity="exact" if args.matching_run else "external",
        budgets=BudgetsV2(
            total_seconds=args.seconds,
            solver_seconds=args.stage_seconds,
            verification_seconds=args.stage_seconds,
            memory_mb=args.memory_mb,
        ),
    )
    objective = make_objective(
        problem.objects,
        profile=(
            ("edit", 1.0),
            ("delete", 1.0),
            ("ontology_edit", 2.0),
            ("human_authored_ontology_edit", 2.0),
        ),
    )
    return problem, objective


def main(argv: Sequence[str] | None = None) -> int:
    """Load bounded repair input, write replay artifacts, and report factored status."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--alignment", type=Path, help="JSON, TSV, CSV or OAEI RDF alignment")
    inputs.add_argument("--matching-run", type=Path, help="completed Exact matching run")
    inputs.add_argument("--problem", type=Path, help="v2 repair input and objective bundle")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--evidence", type=Path, help="optional full deployment feature JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--stage-seconds", type=float, default=10.0)
    parser.add_argument(
        "--model", type=Path, help="frozen XR-2 model checkpoint; enables learned proposals"
    )
    parser.add_argument(
        "--proposal-arm",
        choices=(
            "grammar_mixture",
            "grammar_product",
            "grammar_uniform",
            "bounded_enumeration",
            "rejection",
        ),
        default="grammar_mixture",
    )
    parser.add_argument("--proposal-seconds", type=float, default=30.0)
    parser.add_argument("--compile-seconds", type=float, default=10.0)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--grammar-depth", type=int, default=2)
    parser.add_argument("--constructors", type=int, default=2)
    parser.add_argument("--memory-mb", type=float, help="sampled Linux worker-tree RSS cap")
    args = parser.parse_args(argv)
    if not args.problem and (not args.source or not args.target):
        parser.error("--source and --target are required for alignment inputs")

    from exact.repair.api import write_artifact
    from exact.repair.kernel import repair
    from exact.repair.records import BudgetsV2
    from exact.repair.workers import bounded_call

    BudgetsV2(
        total_seconds=args.seconds,
        solver_seconds=args.stage_seconds,
        verification_seconds=args.stage_seconds,
        memory_mb=args.memory_mb,
    )
    started = time.monotonic()
    prepared = bounded_call(
        _prepare_input,
        args,
        timeout=min(args.seconds, args.stage_seconds),
        memory_mb=args.memory_mb,
    )
    if prepared.status != "complete":
        write_artifact(
            args.output,
            {
                "schema": "exact-repair/run/v2",
                "logical_status": "UNKNOWN",
                "search_status": "UNRESOLVED",
                "stage": "preparation",
                "failure": prepared.detail,
            },
        )
        print(
            json.dumps(
                {
                    "logical_status": "UNKNOWN",
                    "search_status": "UNRESOLVED",
                    "failure": prepared.detail,
                    "output": str(args.output),
                }
            )
        )
        return 2
    problem, objective = prepared.value
    preparation_seconds = time.monotonic() - started
    proposal_seconds = 0.0
    proposal_failure = ""
    if args.model:
        from exact.repair.pipeline import freeze_checkpoint

        before = time.monotonic()
        available = args.seconds - (before - started)
        reserve = min(args.stage_seconds, available / 2)
        proposed = bounded_call(
            freeze_checkpoint,
            problem,
            str(args.model),
            timeout=min(args.proposal_seconds, available - reserve),
            memory_mb=args.memory_mb,
            compile_seconds=args.compile_seconds,
            proposal_arm=args.proposal_arm,
            candidate_cap=args.candidate_cap,
            draws_per_object=args.draws,
            max_depth=args.grammar_depth,
            max_constructors=args.constructors,
            profile=objective.profile,
        )
        proposal_seconds = time.monotonic() - before
        if proposed.status != "complete":
            proposal_failure = f"proposal: {proposed.status}: {proposed.detail}"
            problem = dataclasses.replace(
                problem,
                candidate_coverage="captured_pool_fallback",
                model_status="proposal unavailable; captured frozen objective control",
                proposal_provenance=(
                    {"stage": "proposal", "status": proposed.status, "detail": proposed.detail},
                ),
            )
        else:
            problem, objective = proposed.value.problem, proposed.value.objective
    remaining = args.seconds - (time.monotonic() - started)
    if remaining <= 0:
        write_artifact(
            args.output,
            {
                "schema": "exact-repair/run/v2",
                "input": problem.to_dict(),
                "logical_status": "UNKNOWN",
                "search_status": "UNRESOLVED",
                "stage": "budget",
                "failure": "total repair budget exhausted before selection",
            },
        )
        print(
            json.dumps(
                {
                    "logical_status": "UNKNOWN",
                    "search_status": "UNRESOLVED",
                    "output": str(args.output),
                }
            )
        )
        return 2
    problem = dataclasses.replace(
        problem,
        budgets=dataclasses.replace(
            problem.budgets,
            total_seconds=min(problem.budgets.total_seconds, remaining),
            solver_seconds=min(problem.budgets.solver_seconds, args.stage_seconds),
            verification_seconds=min(problem.budgets.verification_seconds, args.stage_seconds),
            memory_mb=args.memory_mb if args.memory_mb is not None else problem.budgets.memory_mb,
        ),
    )
    result = repair(problem, objective)
    if proposal_failure:
        result = dataclasses.replace(result, failures=(proposal_failure, *result.failures))
    # Serialization is a separately bounded stage; the complete verified result
    # remains owned by the parent if persistence fails.
    payload = {
        "schema": "exact-repair/run/v2",
        "input": problem.to_dict(),
        "objective": objective.to_dict(),
        "result": result.to_dict(),
        "stage_seconds": {
            "preparation": preparation_seconds,
            "proposal": proposal_seconds,
            "selection": result.elapsed_seconds,
        },
    }
    saved = bounded_call(write_artifact, args.output, payload, timeout=args.stage_seconds)
    if saved.status != "complete":
        print(
            json.dumps(
                {
                    "logical_status": result.logical_status,
                    "search_status": result.search_status,
                    "failure": f"serialization: {saved.detail}",
                }
            )
        )
        return 2
    print(
        json.dumps(
            {
                "logical_status": result.logical_status,
                "search_status": result.search_status,
                "verification_scope": result.verification_scope,
                "gap": result.gap,
                "output": str(args.output),
            }
        )
    )
    return 0 if result.assignment is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
