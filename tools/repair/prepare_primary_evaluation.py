"""Freeze disabled primary execution shards and a conservative fit/DEV proposal."""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections import Counter
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.corrective_campaign import source_identity
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import SCHEMA, compile_rows
from tools.repair.primary_runtime import DEV_GPU
from tools.repair.shared_release import bound, immutable, validate_completion


def endpoint_projection(engineering, process_seconds, ledger, dev, *, now, repair_seconds=30):
    """A planning ceiling, never a gate pass inferred from incomplete measurements.

    Amortize the unseparated setup/checkpoint cost over measured updates and
    retain it in every projected epoch; assign the slowest observed update
    to every unmeasured cached case. All six models use the slowest permitted
    endpoint. Combined-loss/service measurements must still establish admission.
    """
    cases, updates = engineering["cases"], engineering["update_seconds"]
    if len(cases) != 128 or not updates or any(not math.isfinite(x) or x <= 0 for x in updates):
        raise ValueError("Expected all TRAIN engineering rows and finite timings")
    cached = sum(r["cache_available"] for r in cases)
    measured = sum(r["optimizer_updates"] for r in cases)
    if measured != len(updates) or measured > cached:
        raise ValueError("Engineering update denominator differs")
    setup = max(0, process_seconds - sum(updates))
    # The trainer checkpoints after every optimizer update, outside the measured
    # update timer. The process/update gap is not a one-time setup measurement.
    overhead_per_update = setup / measured
    epoch_seconds = process_seconds + (cached - measured) * (max(updates) + overhead_per_update)
    stage = ledger["stages"]["primary_training"]
    primary_remaining = max(
        0,
        ledger["stage_limits"]["primary_training"]["gpu_seconds"]
        - stage["cumulative"]["gpu_seconds"],
    )
    learning_start = ledger["stages"]["learning"]["started_epoch"]
    learning_remaining = max(
        0,
        min(
            ledger["stage_limits"]["learning"]["deadline_epoch"],
            learning_start + ledger["stage_limits"]["learning"]["elapsed_seconds"],
        )
        - now,
    )
    rows, models = copy.deepcopy(dev["rows"]), []
    for row in rows:
        row["repair_seconds"] = repair_seconds
        row["case_seconds"] = repair_seconds + row["annotation_reserve_seconds"]
    for seed in (13, 37, 73):
        for condition in ("symbolic", "symbolic_plus_llm"):
            model_id = f"model-{condition}-{seed}"
            model_stage = ledger.get("stages", {}).get(model_id, {})
            charged = model_stage.get("cumulative", {}).get("gpu_seconds", 0)
            model_remaining = max(0, ledger["stage_limits"][model_id]["gpu_seconds"] - charged)
            selected = [
                r
                for r in rows
                if r["selection_slot"]["seed"] == seed
                and r["selection_slot"]["supervision_condition"] == condition
            ]
            reserve = sum(r["case_seconds"] for r in selected)
            # Three owned fit/DEV boundaries reload/checkpoint state; retain a
            # complete measured setup envelope at each boundary, plus cleanup.
            overhead = 3 * setup + 30
            budget = min(
                0.7 * model_remaining, 0.7 * primary_remaining / 6, 0.7 * learning_remaining / 6
            )
            epochs = min(50, max(0, math.floor((budget - reserve - overhead) / epoch_seconds)))
            models.append(
                dict(
                    model_id=model_id,
                    remaining_gpu_seconds=model_remaining,
                    capacity_at_70_percent_seconds=budget,
                    development_reserve_seconds=reserve,
                    setup_checkpoint_cleanup_reserve_seconds=overhead,
                    maximum_projected_epochs=epochs,
                )
            )
    endpoint = min(r["maximum_projected_epochs"] for r in models)
    # Two distinct predeclared passes require an endpoint strictly beyond epoch5.
    endpoint = endpoint if endpoint > 5 else None
    old_final = dev["final_epoch"]
    if endpoint is not None:
        for row in rows:
            old_slot = row["slot_id"]
            row["predecessor_slot_id"] = old_slot
            if row["selection_slot"]["epoch"] == old_final:
                row["selection_slot"]["epoch"] = endpoint
            row["slot_id"] = canonical_hash(row["selection_slot"])
    for model in models:
        model["final_development_reserve_seconds"] = sum(
            r["case_seconds"]
            for r in rows
            if r["selection_slot"]["seed"] == int(model["model_id"].rsplit("-", 1)[1])
            and r["selection_slot"]["supervision_condition"] == model["model_id"].split("-", 2)[1]
            and r["selection_slot"]["epoch"] == (endpoint or old_final)
        )
    return dict(
        schema="exact-repair/primary-endpoint-projection/v1",
        execution_authorized=False,
        endpoint_admitted=False,
        candidate_epochs=endpoint,
        candidate_updates=endpoint * cached if endpoint else None,
        development_epochs=[5, endpoint] if endpoint else None,
        models=models,
        observed_cases=measured,
        cached_cases=cached,
        missing_cache_cases=128 - cached,
        unmeasured_cached_cases=cached - measured,
        epoch_projection_seconds=epoch_seconds,
        unseparated_setup_checkpoint_seconds=setup,
        amortized_setup_checkpoint_seconds_per_update=overhead_per_update,
        checkpoint_overhead_recurs_each_epoch=True,
        primary_remaining_gpu_seconds=primary_remaining,
        learning_remaining_elapsed_seconds=learning_remaining,
        repair_seconds=repair_seconds,
        development_rows=rows,
        unique_semantic_slots=sum(r["semantic_scheduled"] for r in rows),
        independent_swapped_calls=sum(r["swapped"] for r in rows),
        caveat="Slowest-observed imputation is a conservative scenario, not an upper bound or measured full epoch; combined loss and teacher service remain unqualified",
        gates=[
            "qualified pinned teacher and measured service",
            "shared grounded weak labels and TRAIN-only scale",
            "combined-loss throughput/full-inventory projection",
            "matched DEV repair-limit validation",
        ],
        counters_reset=False,
        old_results_rewritten=False,
    )


def prepare(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    registry = read(campaign / "supervisor/registry.json")
    predecessor = registry["primary_runtime_preparation"]["preparation"]
    previous = bound(predecessor)
    schedule = bound(previous["evaluation_schedule"])
    rows = compile_rows(schedule)
    review_ref = registry["primary_runtime_preparation"]["throughput_review"]
    review = bound(review_ref)
    receipts = {}
    reports = {}
    for name, receipt in review["completed_receipts"].items():
        completion, outputs, _, _ = validate_completion(receipt)
        if name in review["report_bindings"]:
            reference = review["report_bindings"][name]
            if (
                outputs.get(str(Path(reference["path"]).relative_to(completion["work"])))
                != reference["sha256"]
            ):
                raise ValueError("Engineering report lacks a nonce-bound output digest")
            reports[name] = bound(reference)
        receipts[name] = receipt
    common = bound(previous["predecessor"])
    dev = bound(common["development_schedule"])
    ledger = read(campaign / "ledger.json")
    immutable(output / "ledger-observation.json", ledger)
    observed_at = time.time()
    engineering = reports["qualify-current-inventory-5090-profile-recovery-001"]
    projection = endpoint_projection(
        engineering, review["process_elapsed_seconds"], ledger, dev, now=observed_at
    )
    projection.update(
        observed_at_epoch=observed_at,
        engineering=review["report_bindings"][
            "qualify-current-inventory-5090-profile-recovery-001"
        ],
        ledger=binding(output / "ledger-observation.json"),
        predecessor_schedule=common["development_schedule"],
        teacher_incident_id=review["teacher_incident_id"],
    )
    immutable(output / "endpoint-projection.json", projection)
    successor_dev = copy.deepcopy(dev)
    successor_dev.update(
        rows=projection["development_rows"],
        repair_seconds=projection["repair_seconds"],
        final_epoch=projection["candidate_epochs"] or dev["final_epoch"],
        case_worker_seconds=sum(r["case_seconds"] for r in projection["development_rows"]),
        schedule_status="candidate_30second_repair_pending_qualified_service_and_paired_throughput",
        execution_authorized=False,
    )
    successor_dev["frozen_annotation_slots"] = {
        r["slot_id"]: dict(selection_slot=r["selection_slot"], swapped=r["swapped"])
        for r in successor_dev["rows"]
        if r["semantic_scheduled"]
    }
    immutable(output / "development-schedule.json", successor_dev)
    source = source_identity()
    immutable(output / "source.json", source)
    protocols = []
    for ref, model in zip(common["protocols"], projection["models"]):
        protocol = bound(ref)
        protocol["identity"].update(
            execution_authorized=False,
            code_hash=source["code_hash"],
            dirty_hash=source["dirty_hash"],
            run_id=output.name + "-" + model["model_id"],
        )
        training = protocol["training"]
        if projection["candidate_epochs"]:
            training["max_epochs"] = projection["candidate_epochs"]
            training["development_epochs"] = projection["development_epochs"]
        training["final_development_reserve_seconds"] = model["final_development_reserve_seconds"]
        own_rows = [
            r
            for r in successor_dev["rows"]
            if r["selection_slot"]["seed"] == training["seeds"][0]
            and r["selection_slot"]["supervision_condition"] == model["model_id"].split("-", 2)[1]
        ]
        training["development_case_seconds"] = {r["slot_id"]: r["case_seconds"] for r in own_rows}
        protocol["llm_labels"]["post_decode_annotation_manifest"] = str(
            output / "development-schedule.json"
        )
        from exact.repair.protocol import RepairProtocolV3

        protocol = RepairProtocolV3.model_validate(protocol).model_dump(by_alias=True)
        # Existing preparation protocols remain historical; these successors are
        # unresolved templates and cannot be used to admit a primary optimizer.
        path = output / "protocols" / Path(ref["path"]).name
        immutable(path, protocol)
        protocols.append(binding(path))
    manifests = []
    # Each case owns its pool and consumers in one shard. No shared output races.
    for cohort in ("conference", "generated"):
        cases = [c for c in schedule["cases"] if c["cohort"] == cohort]
        for start in range(0, len(cases), 4):
            chunk = cases[start : start + 4]
            ids = {c["case_id"] for c in chunk}
            selected = [r for r in rows if r["case_id"] in ids]
            manifest = dict(
                schema=SCHEMA,
                scope="primary_test",
                execution_authorized=False,
                status="disabled_pending_model_freeze_and_public_input_materialization",
                source_commit=source["revision"],
                schedule=previous["evaluation_schedule"],
                rows=selected,
                cases={c["case_id"]: c for c in chunk},
                public_inputs={},
                models={},
                model_freeze_receipt=None,
                gpu_uuid=DEV_GPU,
                device="cuda",
                memory_mb=20000,
                cpus=3,
                ledger_path=str(campaign / "ledger.json"),
                not_before_epoch=schedule["models_freeze_epoch"],
                deadline_epoch=schedule["measurements_deadline_epoch"],
                final_annotation_aggregation_reserve_seconds=28800,
                no_evaluator_access=True,
                hosted_calls=0,
                pending_gates=[
                    "six frozen selected models and matched protocols",
                    "post-freeze public inputs and explicit observable query/score coverage",
                    "evaluation stage reservation",
                ],
            )
            path = output / "execution" / f"{cohort}-{start//4:02d}.json"
            immutable(path, manifest)
            manifests.append(binding(path))
    total_reserved = sum(r["case_seconds"] for r in rows) + 30 * len(manifests)
    capacity = dict(
        schema="exact-repair/evaluation-execution-capacity/v1",
        execution_authorized=False,
        gpu_owner="one exclusive2080 worker per shard",
        reserved_case_seconds=sum(r["case_seconds"] for r in rows),
        shard_setup_cleanup_seconds=30 * len(manifests),
        total_reserved_seconds=total_reserved,
        repair_window_seconds=40 * 3600,
        single_gpu_repair_window_at_70_percent_seconds=0.7 * 40 * 3600,
        projection_passed=total_reserved <= 0.7 * 40 * 3600,
        original_schedule_two_worker_assumption_retained=True,
        pending="Qualify CPU controls/shared-pool ownership or serialized inference service before admitting the original two-worker schedule; no silent second GPU assumption",
        last_eight_hours_reserved=True,
    )
    immutable(output / "execution-capacity.json", capacity)
    result = dict(
        schema="exact-repair/primary-evaluation-execution-preparation/v1",
        status="prepared_not_queued",
        source_commit=source["revision"],
        source=binding(output / "source.json"),
        predecessor=predecessor,
        schedule=previous["evaluation_schedule"],
        dependency_receipts=receipts,
        throughput_review=review_ref,
        endpoint_projection=binding(output / "endpoint-projection.json"),
        execution_capacity=binding(output / "execution-capacity.json"),
        development_schedule=binding(output / "development-schedule.json"),
        protocols=protocols,
        execution_manifests=manifests,
        expected_rows=len(rows),
        comparisons=len(schedule["comparisons"]),
        semantic_slots=128,
        independent_swaps=26,
        annotation_attempts=154,
        execution_authorized=False,
        test_payloads_opened=False,
        hosted_calls=0,
        primary_optimizer_updates=0,
        teacher_incident_id=review["teacher_incident_id"],
        steps=[],
    )
    immutable(output / "prepared.json", result)
    return result


def audit(prepared_path, output):
    value = read(prepared_path)
    schedule = bound(value["schedule"])
    rows = compile_rows(schedule)
    actual = [r for ref in value["execution_manifests"] for r in bound(ref)["rows"]]
    if sorted(actual, key=lambda r: r["id"]) != sorted(rows, key=lambda r: r["id"]):
        raise ValueError("Execution shards differ from the complete frozen schedule")
    for ref in value["execution_manifests"]:
        manifest = bound(ref)
        if manifest["execution_authorized"] or manifest["public_inputs"] or manifest["models"]:
            raise ValueError("Preparation unexpectedly admits TEST inputs or models")
    for receipt in value["dependency_receipts"].values():
        validate_completion(receipt)
    projection = bound(value["endpoint_projection"])
    review = bound(value["throughput_review"])
    expected = endpoint_projection(
        bound(projection["engineering"]),
        review["process_elapsed_seconds"],
        bound(projection["ledger"]),
        bound(projection["predecessor_schedule"]),
        now=projection["observed_at_epoch"],
        repair_seconds=projection["repair_seconds"],
    )
    if any(projection[k] != v for k, v in expected.items()):
        raise ValueError("Endpoint projection changed")
    result = dict(
        schema="exact-repair/primary-evaluation-execution-audit/v1",
        status="complete",
        preparation=binding(Path(prepared_path)),
        expected_rows=len(rows),
        comparisons=len(schedule["comparisons"]),
        operations=dict(Counter(r["adapter"]["operation"] for r in rows)),
        execution_shards=len(value["execution_manifests"]),
        semantic_slots=128,
        conference_semantic_slots=43,
        conference_control_contrasts=1,
        independent_swaps=26,
        annotation_attempts=154,
        candidate_epochs=projection["candidate_epochs"],
        endpoint_admitted=False,
        execution_capacity_admitted=bound(value["execution_capacity"])["projection_passed"],
        test_payloads_opened=False,
        hosted_calls=0,
        primary_optimizer_updates=0,
    )
    immutable(Path(output) / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else audit)(args.input, args.output)


if __name__ == "__main__":
    main()
