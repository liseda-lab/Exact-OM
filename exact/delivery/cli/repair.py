"""Opt-in standalone repair and repair of existing matching run outputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence


def _search_options_record(options: dict[str, Any], risk: Any) -> dict[str, Any]:
    """Freeze scheduling independently of CLI defaults and bind its risk descriptor."""
    from exact.repair.records import canonical_hash

    schema = "exact-repair/search-options/v3"
    values = {
        **options,
        "risk_identity": risk.risk_identity if risk is not None else "none",
        "risk_descriptor_hash": risk.to_dict()["hash"] if risk is not None else "none",
    }
    return {"schema": schema, "hash": canonical_hash((schema, values)), "options": values}


def _resolve_search_options(
    args: argparse.Namespace, payload: dict[str, Any], risk: Any, ledger: Any = None
) -> dict[str, Any]:
    """Restore the saved scheduling epoch and reject conflicting resume overrides."""
    from exact.repair.records import canonical_hash

    names = ("shortlist_size", "utility_window", "shortlist_seconds")
    saved = payload.get("search_options")
    if saved is not None:
        schema = "exact-repair/search-options/v3"
        if (
            not isinstance(saved, dict)
            or set(saved) != {"schema", "hash", "options"}
            or saved["schema"] != schema
        ):
            raise ValueError("invalid frozen search options envelope")
        values = saved["options"]
        if (
            not isinstance(values, dict)
            or set(values) != {*names, "risk_identity", "risk_descriptor_hash"}
            or canonical_hash((schema, values)) != saved["hash"]
        ):
            raise ValueError("frozen search options content hash mismatch")
        if args.resume_search or not args.model:
            if values["risk_identity"] != (
                risk.risk_identity if risk is not None else "none"
            ) or values["risk_descriptor_hash"] != (
                risk.to_dict()["hash"] if risk is not None else "none"
            ):
                raise ValueError("frozen search options risk descriptor mismatch")
        defaults = {name: values[name] for name in names}
    elif args.resume_search:
        if ledger.risk_identity != "none":
            raise ValueError("saved learned search is missing frozen search options")
        defaults = dict(
            shortlist_size=ledger.shortlist_size,
            utility_window=ledger.utility_window,
            shortlist_seconds=ledger.shortlist_seconds,
        )
    else:
        defaults = dict(shortlist_size=1, utility_window=0, shortlist_seconds=10.0)
    result = {}
    for name in names:
        explicit = getattr(args, name)
        if args.resume_search and explicit is not None and explicit != defaults[name]:
            raise ValueError(f"--{name.replace('_', '-')} differs from the frozen search setting")
        result[name] = defaults[name] if explicit is None else explicit
    if (
        type(result["shortlist_size"]) is not int
        or result["shortlist_size"] < 1
        or type(result["utility_window"]) is not int
        or result["utility_window"] < 0
    ):
        raise ValueError("invalid frozen shortlist size/window")
    deadline = result["shortlist_seconds"]
    if deadline is not None and (
        type(deadline) not in (float, int) or not math.isfinite(deadline) or deadline <= 0
    ):
        raise ValueError("invalid frozen shortlist deadline")
    if ledger is not None and (
        result["shortlist_size"],
        result["utility_window"],
        result["shortlist_seconds"],
        risk.risk_identity if risk is not None else "none",
    ) != (
        ledger.shortlist_size,
        ledger.utility_window,
        ledger.shortlist_seconds,
        ledger.risk_identity,
    ):
        raise ValueError("ledger scheduling identity differs from the frozen search")
    return result


def _prepare_input(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load and construct records inside the supervised preparation worker."""
    from exact.repair.api import matching_run_evidence, prepare_repair
    from exact.repair.records import (
        BudgetsV2,
        ObjectiveV2,
        ObjectiveV3,
        RepairInputV2,
        canonical_hash,
        make_objective,
        promote_input_v3,
        read_record,
    )

    if args.problem:
        payload = json.loads(args.problem.read_text())
        problem, objective = read_record(payload["input"]), read_record(payload["objective"])
        if not isinstance(problem, RepairInputV2) or not isinstance(objective, ObjectiveV2):
            raise ValueError("--problem requires versioned input and objective records")
        if not args.resume_search:
            problem = promote_input_v3(problem)
            objective = ObjectiveV3(
                **{
                    **{f.name: getattr(objective, f.name) for f in dataclasses.fields(objective)},
                    "pool_hash": canonical_hash(problem.objects),
                }
            )
        from exact.repair.records import SearchLedgerV3

        ledger: SearchLedgerV3 | None = None
        if args.resume_search:
            loaded_ledger = read_record(json.loads(args.search_ledger.read_text()))
            if not isinstance(loaded_ledger, SearchLedgerV3) or (
                loaded_ledger.input_hash,
                loaded_ledger.objective_hash,
                loaded_ledger.policy_hash,
            ) != (problem.content_hash, objective.content_hash, problem.policy.content_hash):
                raise ValueError("ledger epoch differs from the saved input/objective")
            ledger = loaded_ledger
        risk = None
        if payload.get("risk_descriptor") is not None and not args.model:
            from exact.repair.pipeline import FrozenPlanRisk

            risk = FrozenPlanRisk.from_dict(
                payload["risk_descriptor"], problem=problem, objective=objective
            )
        if ledger is not None and ledger.risk_identity != "none" and risk is None:
            raise ValueError("saved learned search requires its frozen risk descriptor")
        return problem, objective, risk, _resolve_search_options(args, payload, risk, ledger)
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
    return problem, objective, None, _resolve_search_options(args, {}, None)


def main(argv: Sequence[str] | None = None) -> int:
    """Load bounded repair input, write replay artifacts, and report factored status."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--alignment", type=Path, help="JSON, TSV, CSV or OAEI RDF alignment")
    inputs.add_argument("--matching-run", type=Path, help="completed Exact matching run")
    inputs.add_argument("--problem", type=Path, help="v2/v3 repair input and objective bundle")
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
    parser.add_argument(
        "--shortlist-size", type=int, help="default 1; restored from saved run on resume"
    )
    parser.add_argument(
        "--utility-window", type=int, help="default 0; restored from saved run on resume"
    )
    parser.add_argument(
        "--shortlist-seconds", type=float, help="default 10; restored from saved run on resume"
    )
    parser.add_argument("--compiler-cache", type=Path)
    parser.add_argument(
        "--vtree", choices=("balanced", "right_linear", "grouped"), default="balanced"
    )
    parser.add_argument("--search-ledger", type=Path, help="durable v3 parent search state")
    parser.add_argument(
        "--resume-search", action="store_true", help="resume the exact saved input/objective epoch"
    )
    args = parser.parse_args(argv)
    if not args.problem and (not args.source or not args.target):
        parser.error("--source and --target are required for alignment inputs")

    if args.resume_search and (not args.problem or args.model or not args.search_ledger):
        parser.error(
            "--resume-search requires --problem and --search-ledger, with the saved generated pool"
        )

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
                "schema": "exact-repair/run/v3",
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
    problem, objective, risk_scorer, search_options = prepared.value
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
            compiler_cache_directory=str(args.compiler_cache) if args.compiler_cache else None,
            vtree_type=args.vtree,
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
            risk_scorer = proposed.value.risk_scorer
    remaining = args.seconds - (time.monotonic() - started)
    if remaining <= 0:
        write_artifact(
            args.output,
            {
                "schema": "exact-repair/run/v3",
                "input": problem.to_dict(),
                "objective": objective.to_dict(),
                "risk_descriptor": risk_scorer.to_dict() if risk_scorer is not None else None,
                "search_options": _search_options_record(search_options, risk_scorer),
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
    if not args.resume_search:
        problem = dataclasses.replace(
            problem,
            budgets=dataclasses.replace(
                problem.budgets,
                total_seconds=min(problem.budgets.total_seconds, remaining),
                solver_seconds=min(problem.budgets.solver_seconds, args.stage_seconds),
                verification_seconds=min(problem.budgets.verification_seconds, args.stage_seconds),
                memory_mb=(
                    args.memory_mb if args.memory_mb is not None else problem.budgets.memory_mb
                ),
            ),
        )
    result = repair(
        problem,
        objective,
        **search_options,
        risk_order=risk_scorer,
        risk_identity=(risk_scorer.risk_identity if risk_scorer is not None else "none"),
        ledger_path=args.search_ledger,
        resume=args.resume_search,
    )
    if proposal_failure:
        result = dataclasses.replace(result, failures=(proposal_failure, *result.failures))
    # Serialization is a separately bounded stage; the complete verified result
    # remains owned by the parent if persistence fails.
    payload = {
        "schema": "exact-repair/run/v3",
        "input": problem.to_dict(),
        "objective": objective.to_dict(),
        "risk_descriptor": risk_scorer.to_dict() if risk_scorer is not None else None,
        "search_options": _search_options_record(search_options, risk_scorer),
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
