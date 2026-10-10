"""Input-only primary repair adapter; independent assessment is a later stage.

Compilation/audit reads declarations only. Execution requires a distinct admitted
manifest and bound public inputs. No evaluator or hosted client is imported.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import math
import os
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding
from tools.repair.prepare_primary_runtime import validate_schedule
from tools.repair.shared_release import bound, immutable

SCHEMA = "exact-repair/primary-evaluation-execution/v1"
CLEANUP_SECONDS = 5.0


def compile_rows(schedule):
    """Resolve every arm without reading a single referenced TEST payload."""
    validate_schedule(schedule)
    rows = []
    for row in schedule["expected_runs"]:
        arm, experiment, settings = row["arm"], row["experiment"], row["settings"]
        model = settings.get("checkpoint")
        if experiment == "E2_pool":
            operation = "pool"
        elif experiment in {"E2", "E4"} and model:
            operation = "fixed_learned"
        elif experiment == "E3":
            operation = "proposal_support"
            model = settings["weights"] if arm == "learned_circuit" else None
        elif model:
            operation = "generated_learned"
        else:
            operation = {
                "native_deletion": "native_deletion",
                "observable_score_greedy": "observable_score_greedy",
                "observable_local_support": "support_retention_surrogate",
                "budgeted_complete_plan_query": "complete_plan_observable_query",
                "deletion_after_rich": "deletion_after_rich",
            }[arm]
        rows.append(
            dict(
                row,
                adapter=dict(
                    operation=operation,
                    model=model,
                    pool_dependency=row["pool"],
                    cost_identity=row["id"],
                    hosted_calls=0,
                    evaluator_access=False,
                    output_directory="rows/" + row["id"],
                ),
            )
        )
    by_id = {r["id"]: r for r in rows}
    for row in rows:
        if row["pool"]:
            pool = by_id[row["pool"]]
            if pool["adapter"]["operation"] != "pool" or any(
                pool[k] != row[k] for k in ("case_id", "input_identity", "protocol")
            ):
                raise ValueError("Shared pool identity differs")
    # E4/E5 reuse is permitted only for the exact original row and matched knobs.
    for comparison in schedule["comparisons"]:
        left, right = (by_id[comparison[k]] for k in ("left", "right"))
        if any(
            left[k] != right[k] for k in ("case_id", "input_identity", "case_seconds", "protocol")
        ):
            raise ValueError("Comparison input/budget mismatch")
        if comparison["experiment"] == "E4":
            keys = ("checkpoint", "objective_and_cuts", "generation")
            if left["pool"] != right["pool"] or any(
                left["settings"].get(k) != right["settings"].get(k) for k in keys
            ):
                raise ValueError("E4 changed objective, pool or cuts")
        if comparison["experiment"] == "E5":
            if any(
                left["settings"].get(k) != right["settings"].get(k)
                for k in ("checkpoint", "generation", "risk")
            ):
                raise ValueError("E5 changed model or language")
    return rows


def remaining(deadline, cap=None):
    left = min(deadline, float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf"))) - time.time()
    if cap is not None:
        left = min(left, cap)
    if left <= 0:
        raise TimeoutError("Primary evaluation case/stage deadline exhausted")
    return left


def _call(function, *args, deadline, cap=None, memory_mb=8192, **kwargs):
    from exact.repair.workers import bounded_call

    outcome = bounded_call(
        function, *args, timeout=remaining(deadline, cap), memory_mb=memory_mb, **kwargs
    )
    if not outcome.cleanup_complete:
        raise RuntimeError("Nested evaluation cleanup incomplete")
    if outcome.status == "error":
        raise RuntimeError(outcome.detail)
    return outcome


def uniform_pool(problem, protocol, directory, *, decoder="circuit", seed=13, device="cpu"):
    """Both decoders use one zero-logit component and canonical alias mass."""
    import torch

    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from tools.repair.common_inventory import observable_graph
    from tools.repair.evaluate_campaign import generation_options

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    _, graph = observable_graph(problem, protocol, directory)
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
    model.to(device)
    model.eval()
    options = generation_options(protocol, Path(directory) / "compiler-cache")
    options.update(seed=seed, mixtures=1)
    return freeze_neural_round(
        problem,
        model,
        graph=graph,
        proposal_arm="grammar_uniform" if decoder == "circuit" else "matched_grammar_decoder",
        **options,
    )


def _public_problem(public, case):
    from tools.repair.corrective_study import PUBLIC_INPUT_SCHEMA

    if public.get("schema") != PUBLIC_INPUT_SCHEMA:
        raise ValueError("Only bound public input records are accepted")
    metadata = public["metadata"]
    if any(metadata.get(k) != case.get(k) for k in ("case_id", "structural_parent", "split")):
        raise ValueError("Public input case/parent/split differs")
    if set(public) - {
        "schema",
        "metadata",
        "input",
        "observable_queries",
        "observable_mapping_scores",
    }:
        raise ValueError("Evaluator fields cannot enter inference")
    problem = read_record(public["input"])
    if case.get("input_hash") and problem.content_hash != case["input_hash"]:
        raise ValueError("Public input hash differs")
    return problem


def evaluate_input(row, case, public_ref, protocol_ref, model, pool, directory, deadline, device):
    """Execute one row using observable inputs, frozen models and native checking."""
    from exact.repair.kernel import repair
    from exact.repair.pipeline import bounded_freeze_checkpoint, repair_neural_round
    from exact.repair.records import make_objective
    from tools.repair import corrective_study as study
    from tools.repair.common_inventory import freeze_inventory
    from tools.repair.evaluate_campaign import generation_options

    started = time.monotonic()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["EXACT_REPAIR_DEADLINE_EPOCH"] = str(
        min(deadline, float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf")))
    )
    public, protocol = bound(public_ref), bound(protocol_ref)
    problem = _public_problem(public, case)
    problem = dataclasses.replace(
        problem,
        budgets=dataclasses.replace(
            problem.budgets,
            total_seconds=remaining(deadline),
            memory_mb=protocol["resources"]["case_rss_mb"],
            verification_seconds=min(
                protocol["resources"]["verification_seconds"], remaining(deadline)
            ),
        ),
    )
    operation, settings = row["adapter"]["operation"], row["settings"]
    queries = public.get("observable_queries", [])
    if (
        operation
        in {"support_retention_surrogate", "complete_plan_observable_query", "proposal_support"}
        and not queries
    ):
        return dict(status="unavailable_observable_queries", logical_status="UNKNOWN")
    if pool:
        shared = read_record(bound(pool))
        if (
            shared.fixed_axioms != problem.fixed_axioms
            or shared.policy != problem.policy
            or [(o.object_id, o.original_axioms) for o in shared.objects]
            != [(o.object_id, o.original_axioms) for o in problem.objects]
        ):
            raise ValueError("Shared inventory changed input or policy")
        problem = dataclasses.replace(shared, budgets=problem.budgets)
    if operation == "generated_learned":
        return study.evaluate_row(
            public,
            dict(
                arm="learned",
                checkpoint=model,
                protocol=protocol,
                seconds=remaining(deadline),
                schedule=(
                    "staged_verified_repair" if settings["execution"] == "staged" else "one_stage"
                ),
                generation_overrides=dict(seed=row["seed"], device=device),
            ),
            directory,
        )
    if operation in {
        "native_deletion",
        "support_retention_surrogate",
        "complete_plan_observable_query",
    }:
        return study.evaluate_row(
            public,
            dict(
                arm=operation,
                protocol=protocol,
                seconds=remaining(deadline),
                observable_queries=queries,
                **(dict(frozen_inventory=pool) if pool else {}),
            ),
            directory,
        )
    frozen = None
    if operation in {"pool", "proposal_support", "deletion_after_rich"}:
        if model:
            generated = bounded_freeze_checkpoint(
                problem,
                model["path"],
                checkpoint_sha256=model["sha256"],
                seconds=remaining(deadline, protocol["resources"]["generation_seconds"]),
                memory_mb=problem.budgets.memory_mb,
                **{
                    **generation_options(protocol, directory / "compiler-cache"),
                    "seed": row["seed"],
                    "device": device,
                },
            )
            if not generated.cleanup_complete:
                raise RuntimeError("Nested generation cleanup incomplete")
            if generated.status == "error":
                raise RuntimeError(generated.detail)
        else:
            generated = _call(
                uniform_pool,
                problem,
                protocol,
                str(directory),
                deadline=deadline,
                cap=protocol["resources"]["generation_seconds"],
                memory_mb=problem.budgets.memory_mb,
                decoder=settings.get("decoder", "circuit"),
                seed=settings.get("sampler_seed", row["seed"] or 13),
                device=device,
            )
        if generated.status != "complete":
            return dict(status="generation_" + generated.status, logical_status="UNKNOWN")
        frozen = generated.value
        problem = dataclasses.replace(frozen.problem, budgets=problem.budgets)
        immutable(directory / "pool.json", problem.to_dict())
        import json

        from exact.repair.records import canonical_json

        immutable(
            directory / "generation.json", json.loads(canonical_json(frozen.proposal_reports))
        )
        if operation == "pool":
            return dict(
                status="pool_complete",
                pool=binding(directory / "pool.json"),
                inventory_hash=canonical_hash(problem.objects),
                logical_status="NOT_APPLICABLE",
            )
        if operation == "proposal_support":
            return study.evaluate_row(
                public,
                dict(
                    arm="support_retention_surrogate",
                    protocol=protocol,
                    seconds=remaining(deadline),
                    observable_queries=queries,
                    frozen_inventory=binding(directory / "pool.json"),
                ),
                directory,
            )
    profile = tuple(sorted(protocol["objective"]["edit_weights"].items()))
    selection_offset = 0.0
    if operation == "fixed_learned":
        frozen = freeze_inventory(
            problem,
            dict(kind="learned", id=row["arm"], model=model, model_hash=model["model_hash"]),
            protocol,
            directory,
            device=device,
        )
        if settings["risk"] == "off":
            frozen = dataclasses.replace(frozen, risk_scorer=None)
        frozen = dataclasses.replace(
            frozen,
            problem=dataclasses.replace(
                problem,
                budgets=dataclasses.replace(problem.budgets, total_seconds=remaining(deadline)),
            ),
        )
        selection_offset = time.monotonic() - started
        result = repair_neural_round(
            frozen,
            shortlist_size=protocol["selection"]["shortlist_size"],
            utility_window=protocol["selection"]["utility_window"],
            shortlist_seconds=min(
                protocol["selection"]["construction_seconds"], remaining(deadline)
            ),
            diagnose=True,
            preserve_verified_input=False,
            ledger_path=directory / "search-ledger.json",
        )
        objective = frozen.objective
    elif operation == "deletion_after_rich":
        # Rich generation is charged, but the subsequent language is keep/delete.
        problem = study.elementary_pool(problem)
        objective = make_objective(
            problem.objects, profile=profile, scale=protocol["objective"]["integer_scale"]
        )
        problem = dataclasses.replace(
            problem, budgets=dataclasses.replace(problem.budgets, total_seconds=remaining(deadline))
        )
        selection_offset = time.monotonic() - started
        result = repair(
            problem, objective, diagnose=True, ledger_path=directory / "search-ledger.json"
        )
    elif operation == "observable_score_greedy":
        scores = public.get("observable_mapping_scores")
        if not scores or any(o.kind != "mapping" for o in problem.objects):
            return dict(status="unavailable_observable_scores", logical_status="UNKNOWN")
        problem = study.elementary_pool(problem)
        if set(scores) != {o.object_id for o in problem.objects} or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in scores.values()
        ):
            raise ValueError("Observable scores must cover exactly the original mappings")
        # Predeclared greedy prefix: successively remove lowest-score mappings;
        # every candidate plan is checked by the same native kernel.
        assignment = [
            next(i for i, c in enumerate(o.candidates) if c.axioms == o.original_axioms)
            for o in problem.objects
        ]
        order = sorted(
            range(len(problem.objects)),
            key=lambda i: (scores[problem.objects[i].object_id], problem.objects[i].object_id),
        )
        result = objective = None
        for index in [None, *order]:
            if index is not None:
                choices = [
                    i for i, c in enumerate(problem.objects[index].candidates) if not c.axioms
                ]
                if not choices:
                    continue
                assignment[index] = choices[0]
            remaining(deadline)
            from exact.repair.records import replace_inventory

            singleton = replace_inventory(
                problem,
                tuple(
                    dataclasses.replace(o, candidates=(o.candidates[i],))
                    for o, i in zip(problem.objects, assignment)
                ),
            )
            singleton = dataclasses.replace(
                singleton,
                budgets=dataclasses.replace(singleton.budgets, total_seconds=remaining(deadline)),
            )
            objective = make_objective(singleton.objects, profile=profile)
            selection_offset = time.monotonic() - started
            result = repair(singleton, objective, diagnose=True, preserve_verified_input=False)
            if result.verification is not None and result.verification.authorizes:
                problem = singleton
                break
        else:
            problem = singleton
    else:
        raise ValueError("Unsupported primary operation")
    immutable(
        directory / "selection.json",
        dict(input=problem.to_dict(), objective=objective.to_dict(), result=result.to_dict()),
    )
    if result.logical_status == "VERIFIED_FEASIBLE" and (
        result.verification is None or not result.verification.authorizes
    ):
        raise ValueError("Verified output lacks a native certificate")
    return dict(
        status="evaluated",
        logical_status=result.logical_status,
        search_status=result.search_status,
        first_verified_seconds=(
            selection_offset + result.first_verified_seconds
            if result.first_verified_seconds is not None
            else None
        ),
        first_verified_timing_scope="row_worker_including_generation",
        checks=result.checks,
        selected_ids=[c.candidate_id for c in result.selected],
        semantic_status="independent_assessment_pending",
    )


def bounded_evaluate_input(*args):
    try:
        return evaluate_input(*args)
    except TimeoutError as error:
        return dict(status="resource_limited", logical_status="UNKNOWN", detail=str(error))


def validate_saved(saved, identity):
    if saved["identity"] != identity:
        raise ValueError("Incompatible evaluation row resume")
    for artifact in saved.get("artifacts", []):
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Evaluation row output checksum changed")
    if not saved.get("cleanup_complete"):
        raise RuntimeError("Prior row requires cleanup reconciliation")


def execute_rows(manifest, output, *, evaluator=bounded_evaluate_input, now=time.time):
    """One exclusive shard owner; terminal scientific unknowns are never replayed."""
    if manifest.get("schema") != SCHEMA or not manifest.get("execution_authorized"):
        raise ValueError("Primary evaluation is not admitted")
    if manifest.get("scope") != "fixture":
        if now() < manifest["not_before_epoch"] or not manifest.get("model_freeze_receipt"):
            raise ValueError("TEST remains closed until the model freeze")
        freeze = bound(manifest["model_freeze_receipt"])
        if (
            freeze.get("models") != manifest["models"]
            or len(manifest["models"]) != 6
            or freeze.get("test_outcomes_opened") is not False
        ):
            raise ValueError("Six selected checkpoints must match the closed-TEST freeze receipt")
        schedule = bound(manifest["schedule"])
        expected = {r["id"]: r for r in compile_rows(schedule)}
        if any(expected.get(r["id"]) != r for r in manifest["rows"]):
            raise ValueError("Execution rows differ from the frozen schedule")
        declared = {c["case_id"]: c for c in schedule["cases"]}
        if any(declared.get(key) != value for key, value in manifest["cases"].items()):
            raise ValueError("Execution case declaration changed")
        from tools.repair.primary_runtime import owned_device

        if "ownership_lane" in manifest:
            from tools.repair.evaluation_ownership import validate_lane

            validate_lane(manifest)
            admission = bound(manifest["capacity_admission"])
            if (
                admission.get("concurrent_load_profile") != manifest["concurrent_load_profile"]
                or admission.get("source_commit") != manifest["source_commit"]
                or not all(
                    admission.get(k) is True
                    for k in (
                        "cpu_native_qualified",
                        "matched_inference_load_qualified",
                        "remaining_stage_projection_passed",
                    )
                )
            ):
                raise ValueError("Evaluation capacity qualification remains unresolved")
        else:
            owned_device(manifest["gpu_uuid"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "owner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = canonical_hash(manifest)
        immutable(output / "identity.json", dict(identity=identity))
        if manifest.get("scope") == "fixture":
            reference = manifest["ledger_snapshot"]
            if sha(reference["path"]) != reference["sha256"]:
                raise ValueError("Fixture ledger binding changed")
            stage = read(reference["path"])
        else:
            stage = read(manifest["ledger_path"])
            from tools.repair.evaluation_ownership import validate_owner

            validate_owner(manifest, stage)
        first = stage.get("stages", {}).get("evaluation", {}).get("started_epoch")
        if first is None:
            raise ValueError("Evaluation first admission must be durably reserved")
        limit = min(
            manifest["deadline_epoch"],
            first + 40 * 3600,
            float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf")),
        )
        from tools.repair.evaluation_ownership import external_pools

        dependencies = external_pools(manifest)
        rows, saved_rows = manifest["rows"], {}
        error = None
        for row in rows:
            directory = output / "rows" / row["id"]
            path, guard = directory / "completion.json", directory / "started.json"
            row_identity = canonical_hash((identity, row))
            if path.exists():
                saved = read(path)
                validate_saved(saved, row_identity)
                saved_rows[row["id"]] = saved
                if saved["status"] == "error":
                    error = RuntimeError(
                        "Prior implementation failure requires a specific recovery"
                    )
                    break
                continue
            if guard.exists():
                started = read(guard)
                if started.get("identity") != row_identity:
                    raise ValueError("Incompatible interrupted row identity")
                saved_rows[row["id"]] = dict(
                    identity=row_identity,
                    status="interrupted_unknown",
                    result=None,
                    cleanup_complete=None,
                    charged_seconds=started["reserved_seconds"],
                    reserved_seconds=started["reserved_seconds"],
                    shared_cost_reference=row["adapter"]["pool_dependency"],
                    cost_rule="Full reservation retained pending owner/ledger reconciliation",
                )
                error = RuntimeError(
                    "Interrupted row requires owner/ledger reconciliation; no renewed case budget"
                )
                break
            if now() + CLEANUP_SECONDS >= limit:
                break
            dependency = row["adapter"]["pool_dependency"]
            pool_result = (
                (saved_rows.get(dependency, dependencies.get(dependency, {})).get("result") or {})
                if dependency
                else {}
            )
            public = manifest["public_inputs"].get(row["case_id"])
            model_name = row["adapter"]["model"]
            model = manifest["models"].get(model_name) if model_name else None
            reason = (
                "unavailable_public_input"
                if public is None
                else (
                    "unavailable_model"
                    if model_name and model is None
                    else (
                        "unavailable_shared_pool"
                        if dependency and not pool_result.get("pool")
                        else None
                    )
                )
            )
            directory.mkdir(parents=True, exist_ok=True)
            if reason:
                saved = dict(
                    identity=row_identity,
                    status=reason,
                    result=None,
                    cleanup_complete=True,
                    charged_seconds=0,
                    artifacts=[],
                    shared_cost_reference=dependency,
                )
            else:
                started = now()
                case_deadline = min(
                    limit - CLEANUP_SECONDS, started + row["case_seconds"] - CLEANUP_SECONDS
                )
                immutable(
                    guard,
                    dict(
                        identity=row_identity,
                        started_epoch=started,
                        deadline_epoch=case_deadline,
                        reserved_seconds=row["case_seconds"],
                        cost_identity=row["id"],
                    ),
                )
                from exact.repair.workers import bounded_call

                outcome = bounded_call(
                    evaluator,
                    row,
                    manifest["cases"][row["case_id"]],
                    public,
                    row["protocol"],
                    model,
                    pool_result.get("pool"),
                    str(directory / "payload"),
                    case_deadline,
                    manifest.get("device", "cpu"),
                    timeout=remaining(case_deadline),
                    memory_mb=manifest["memory_mb"],
                    cpu_seconds=manifest.get("cpus", 3) * remaining(case_deadline),
                )
                saved = dict(
                    identity=row_identity,
                    status=outcome.status,
                    result=outcome.value,
                    detail=outcome.detail,
                    cleanup_complete=outcome.cleanup_complete,
                    charged_seconds=now() - started,
                    reserved_seconds=row["case_seconds"],
                    resources=dict(outcome.resource_usage),
                    shared_cost_reference=dependency,
                    artifacts=[
                        binding(p)
                        for p in sorted((directory / "payload").rglob("*"))
                        if p.is_file() and p.suffix != ".lock"
                    ],
                )
                if outcome.status == "error" or not outcome.cleanup_complete:
                    error = RuntimeError(
                        "Implementation failure or incomplete cleanup: " + outcome.detail
                    )
            immutable(path, saved)
            saved_rows[row["id"]] = saved
            if error:
                break
        # Include every durable result even when an earlier row stops admission.
        for row in rows:
            path = output / "rows" / row["id"] / "completion.json"
            if path.exists() and row["id"] not in saved_rows:
                saved = read(path)
                validate_saved(saved, canonical_hash((identity, row)))
                saved_rows[row["id"]] = saved
        report = dict(
            schema=SCHEMA,
            manifest_identity=identity,
            expected_rows=len(rows),
            rows=[
                dict(
                    id=r["id"], **saved_rows.get(r["id"], dict(status="not_attempted", result=None))
                )
                for r in rows
            ],
            charged_seconds=sum(r["charged_seconds"] for r in saved_rows.values()),
            cost_rule="Each run/pool charged once; comparisons only reference run identities",
            external_pool_cost_references=sorted(dependencies),
            ownership_lane=manifest.get("ownership_lane", "legacy_full_gpu_shard"),
            concurrent_load_profile=manifest.get("concurrent_load_profile"),
            hosted_calls=0,
        )
        write_artifact(output / "report.json", report)
        if error:
            raise error
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    execute_rows(read(args.manifest), args.output)


if __name__ == "__main__":
    main()
