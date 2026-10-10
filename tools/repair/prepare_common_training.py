"""Prepare inert paired protocols and schedules from the closed two-round release.

This command never submits work, changes a registry, fits a model, or makes a
hosted call. A later admission successor must resolve every listed runtime gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sqlite3

from exact.repair.protocol import RepairProtocolV3
from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.common_training import CONDITIONS, SCHEMA, development_schedule, load_release
from tools.repair.corrective_campaign import source_identity
from tools.repair.historical_regression import binding
from tools.repair.shared_release import bound, immutable


def service_observation(campaign):
    """Read only model, status and timing from the existing durable wire ledger."""
    observations = defaultdict(list)
    with sqlite3.connect(
        f"file:{campaign / 'annotations/ledger/requests.sqlite3'}?mode=ro", uri=True
    ) as db:
        for identity, status, elapsed in db.execute(
            "SELECT r.identity,a.status,a.elapsed_seconds FROM requests r JOIN attempts a USING(request_id)"
        ):
            model = json.loads(identity)["payload"]["model"]
            observations[model].append((status, elapsed))
    return dict(
        schema="exact-repair/teacher-service-observation/v1",
        scope="all earlier failed/unavailable attempts included; no profile qualified by this timing",
        models={
            model: dict(
                attempts=len(rows),
                statuses=dict(Counter(str(r[0]) for r in rows)),
                measured_attempts=sum(r[1] is not None for r in rows),
                sum_seconds=sum(r[1] or 0 for r in rows),
                max_seconds=max((r[1] for r in rows if r[1] is not None), default=None),
            )
            for model, rows in sorted(observations.items())
        },
        final_schedule_admission="requires measured service of the eventually qualified pinned teacher",
    )


def native_calibration_observation(campaign, registry):
    """Include every completed or unavailable exposed calibration row and cost."""
    from tools.repair.shared_release import validate_completion

    reports, strata, intervals = [], defaultdict(list), []
    for run_id in ("calibration-study-0", "calibration-study-1", "calibration-study-2"):
        run = next(r for r in registry["runs"] if r["id"] == run_id)
        attempt = Path(run["completion_path"]).parent
        completion = read(attempt / "completion.json")
        receipt = {k: binding(attempt / (k + ".json")) for k in ("completion", "step", "outputs")}
        receipt.update(
            batch=binding(Path(completion["batch"])),
            dispatch_nonce=run["dispatch_nonce"],
            step_id=run["step_id"],
            expected_status="complete",
        )
        completion, outputs, _, _ = validate_completion(receipt)
        report_ref = binding(Path(completion["work"]) / "report.json")
        if outputs.get("report.json") != report_ref["sha256"]:
            raise ValueError("Calibration report is not a terminal output")
        report = bound(report_ref)
        if len(report["rows"]) != report["expected_rows"]:
            raise ValueError("Calibration denominator differs")
        for row in report["rows"]:
            result = row.get("result") or {}
            key = "/".join(str(result.get(k, "unavailable")) for k in ("family", "control", "arm"))
            strata[key].append(
                dict(
                    process=row["status"],
                    logical=result.get("logical_status", "unavailable"),
                    seconds=row.get("resources", {}).get("wall_seconds", row["reserved_seconds"]),
                    rss=row.get("resources", {}).get("peak_sampled_tree_rss_bytes", 0),
                )
            )
        reports.append(dict(report=report_ref, receipt=receipt, scheduled=report["expected_rows"]))
        intervals.append((completion["started_epoch"], completion["finished_epoch"]))
    return dict(
        schema="exact-repair/native-calibration-observation/v1",
        reports=reports,
        all_scheduled_rows=sum(r["scheduled"] for r in reports),
        simultaneous_three_worker_seconds=max(
            0, min(b for a, b in intervals) - max(a for a, b in intervals)
        ),
        strata={
            key: dict(
                rows=len(rows),
                process_statuses=dict(Counter(r["process"] for r in rows)),
                logical_statuses=dict(Counter(r["logical"] for r in rows)),
                timeout_inclusive_seconds=sum(r["seconds"] for r in rows),
                maximum_case_seconds=max(r["seconds"] for r in rows),
                peak_rss_bytes=max(r["rss"] for r in rows),
            )
            for key, rows in sorted(strata.items())
        },
        scope="exposed engineering calibration; no trained-model or target selection evidence",
    )


def prepare(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    registry = read(campaign / "supervisor/registry.json")
    registered = next(
        r for r in registry["runs"] if r["id"] == "audit-common-generated-final-release-001"
    )
    attempt = Path(registered["completion_path"]).parent
    completion = read(attempt / "completion.json")
    audit = {key: binding(attempt / (key + ".json")) for key in ("completion", "step", "outputs")}
    audit.update(
        batch=binding(Path(completion["batch"])),
        dispatch_nonce=registered["dispatch_nonce"],
        step_id=registered["step_id"],
        expected_status="complete",
    )
    release_ref = registry["shared_generated_release"]["final_report"]
    cases, caches, loaded = load_release(release_ref, audit)
    release = bound(release_ref)
    dev = bound(release["development"])["rows"]
    source = source_identity()
    if source["dirty_hash"] != __import__("hashlib").sha256(b"").hexdigest():
        raise ValueError("Commit the tested adapter before preparing successor protocols")
    immutable(output / "source.json", source)
    immutable(output / "service-observation.json", service_observation(campaign))
    immutable(
        output / "native-calibration-observation.json",
        native_calibration_observation(campaign, registry),
    )
    immutable(output / "ledger-observation.json", read(campaign / "ledger.json"))
    immutable(
        output / "annotation-reservations-observation.json",
        read(campaign / "annotations/ledger/phase-reservations.json"),
    )
    approved_manifest = read(campaign / "annotations/calibration-replacement-001/manifest.json")
    amendment = {k: approved_manifest[k] for k in ("request_budget_amendment", "request_limits")}
    for ref in amendment["request_budget_amendment"].values():
        bound(ref)
    schedule = development_schedule(dev)
    schedule.update(
        amendment,
        phase="development",
        authorized=False,
        teacher_profile="UNFROZEN/four-model-panel-selection",
        service_observation=binding(output / "service-observation.json"),
        development_use_policy="development_selection/v1",
        maximum_requests=96,
        maximum_service_seconds=14400,
        maximum_input_tokens=8000,
        maximum_output_tokens=2000,
        independent_test_judge_used=False,
        exact_plan_policy="fresh decoded complete plan plus committed native verification; no nearby-plan labels",
        prompt_version="semantic-fidelity-prompt/v3.2",
        rubric_version="semantic-fidelity/v3",
    )
    immutable(output / "development-schedule.json", schedule)
    train_cases = [c for c in cases if c.split == "train"]
    # Two fixed comparison intentions per original case. Concrete verified plans
    # and evidence are an explicit later prerequisite, not invented here.
    intentions = [
        dict(
            id=canonical_hash((c.case_id, index, "TRAIN-intention/v1")),
            case_id=c.case_id,
            structural_parent=c.structural_parent,
            family=c.family,
            control=c.control,
            comparison_index=index,
            status="pending_exact_grounded_packet",
            swapped=False,
        )
        for c in train_cases
        for index in range(2)
    ]
    audit_ids = {
        r["id"] for r in sorted(intentions, key=lambda r: canonical_hash(("swap", r["id"])))[:52]
    }
    train_shards = []
    for shard_index in range(2):
        rows = intentions[128 * shard_index : 128 * (shard_index + 1)]
        rows = rows + [
            dict(r, id=r["id"] + "-swap", swapped=True, original_id=r["id"])
            for r in rows
            if r["id"] in audit_ids
        ]
        path = output / "annotation-intentions" / f"train-{shard_index:02d}.json"
        immutable(
            path,
            dict(
                schema="exact-repair/annotation-intentions/v1",
                authorized=False,
                phase="train",
                slots=rows,
                scheduled=len(rows),
                maximum_requests=352,
                unique_comparisons=128,
                teacher_profile="UNFROZEN/four-model-panel-selection",
                maximum_input_tokens=8000,
                maximum_output_tokens=2000,
                request_seconds=90,
                maximum_in_flight=4,
                maximum_service_seconds=28800,
                prompt_version="semantic-fidelity-prompt/v3.2",
                rubric_version="semantic-fidelity/v3",
                exact_plans_bound=False,
                grounded_packet_gate_passed=False,
                **amendment,
            ),
        )
        train_shards.append(binding(path))
    fitted = [c.case_id for c in train_cases if c.case_id in caches]
    endpoint = dict(
        status="candidate_pending_measured_paired_admission",
        maximum_epochs=50,
        candidate_common_epoch=50,
        updates_per_epoch=len(fitted),
        candidate_common_updates=50 * len(fitted),
        intermediate_dev_epoch=5,
        final_dev_epoch=50,
        batch_size=1,
        common_fitting_case_ids=fitted,
        full_train_denominator=[c.case_id for c in train_cases],
        missing_training_cache_ids=[c.case_id for c in train_cases if c.case_id not in caches],
        completed_update_rule="both arms must reach the same pre-fit agreed endpoint; do not lower one arm independently",
        incomplete_pair_rule="retain both arms and missing endpoint in denominator; no implicit epoch/final-pass substitution",
        data_order_hashes={},
    )
    for seed in (13, 37, 73):
        orders = []
        for epoch in range(50):
            order = fitted.copy()
            random.Random(seed + epoch).shuffle(order)
            orders.append(canonical_hash(order))
        endpoint["data_order_hashes"][str(seed)] = orders
    immutable(output / "common-endpoint.json", endpoint)
    protocols, preparations = [], []
    paired_invariants = None
    for seed in (13, 37, 73):
        for condition in CONDITIONS:
            protocol = read(campaign / "protocols" / f"hgt-pair-{condition}-s{seed}.json")
            protocol["identity"].update(
                execution_authorized=False,
                code_hash=source["code_hash"],
                dirty_hash=source["dirty_hash"],
                purpose="development",
                run_id=f"common-training-001-{condition}-s{seed}",
            )
            own = [
                r
                for r in schedule["rows"]
                if r["selection_slot"]["seed"] == seed
                and r["selection_slot"]["supervision_condition"] == condition
            ]
            protocol["training"].update(
                sampled_assignments=0,
                max_epochs=50,
                development_epochs=[5, 50],
                max_full_development_evaluations=2,
                patience_enabled=False,
                development_case_ids=[r["case_id"] for r in dev],
                development_case_seconds={r["slot_id"]: r["case_seconds"] for r in own},
                final_development_reserve_seconds=sum(
                    r["case_seconds"] for r in own if r["selection_slot"]["epoch"] == 50
                ),
            )
            protocol["resources"].update(
                allocated_gpus=1,
                concurrency=1,
                case_wall_seconds=392.0,
                campaign_wall_seconds=21600.0,
                campaign_gpu_hours=3.0,
            )
            protocol["resources"]["stage_wall_seconds"]["train"] = 21600.0
            protocol["losses"]["scale_alignment"] = "UNFROZEN/TRAIN-only-weak-scale-calibration"
            protocol["llm_labels"].update(
                teacher_profile="UNFROZEN/four-model-panel-selection",
                post_decode_annotation_manifest=str(output / "development-schedule.json"),
                development_use_policy="development_selection/v1",
            )
            # Numeric placeholders remain visible; the UNFROZEN identity blocks
            # accidental execution until a versioned teacher/scale successor.
            validated = RepairProtocolV3.model_validate(protocol)
            inv = {
                k: protocol[k]
                for k in ("model", "objective", "generation", "collection", "teacher")
            }
            if paired_invariants is not None and inv != paired_invariants:
                raise ValueError("Paired architecture/data generation/costs changed")
            paired_invariants = inv
            path = output / "protocols" / f"hgt-pair-{condition}-s{seed}.json"
            immutable(path, validated.model_dump(by_alias=True))
            protocols.append(binding(path))
            value = dict(
                schema=SCHEMA, release=release_ref, audit_run=audit, protocol=binding(path)
            )
            prep = output / "preparations" / f"hgt-pair-{condition}-s{seed}.json"
            immutable(prep, dict(value, hash=canonical_hash(value)))
            preparations.append(binding(prep))
    # A per-case envelope is not a measured concurrency projection. Expose both.
    immutable(
        output / "admission.json",
        dict(
            execution_authorized=False,
            scheduling_fraction=0.7,
            dev_worker_reservation_seconds=schedule["case_worker_seconds"],
            dev_reasoner_capacity_at_70_percent_seconds=0.7 * 36 * 3600,
            dev_reasoner_inequality_passes=schedule["case_worker_seconds"] <= 0.7 * 36 * 3600,
            dev_hosted_reservation_seconds=(64 + 14) * 90,
            final_dev_reserve_is_remaining_frozen_schedule=True,
            gpu_admission_passed=False,
            final_endpoint_frozen=False,
            unresolved=[
                "qualified pinned TRAIN/DEV teacher and measured service",
                "grounded shared weak labels and TRAIN-only scale",
                "measured matched fit throughput on frozen generated inventories",
                "phase-separated5090 fit/2080DEV orchestration with cumulative model/stage/GPU budgets; monolithic trainer is not admitted",
            ],
            budget_reset=False,
            source_of_limits=binding(campaign / "planning-contract.json"),
        ),
    )
    result = dict(
        schema="exact-repair/common-training-contract-preparation/v1",
        status="prepared_not_queued",
        source=binding(output / "source.json"),
        source_commit=source["revision"],
        common_release=release_ref,
        audit_run=audit,
        coverage=release["coverage"],
        cohort="generated_only",
        missing_real_coverage=True,
        expected_train_cases=128,
        expected_development_cases=32,
        committed_training_caches=len(caches),
        symbolic_labels=sum(len(c.labels) for c in caches.values()),
        protocols=protocols,
        preparations=preparations,
        train_annotation_intentions=train_shards,
        development_schedule=binding(output / "development-schedule.json"),
        endpoint=binding(output / "common-endpoint.json"),
        admission=binding(output / "admission.json"),
        existing_teacher_amendment_incident=registry["lower_cost_teacher_comparison"][
            "incident_id"
        ],
        annotation_attempts_transmitted=0,
        optimizer_updates=0,
        steps=[],
        registry_changes=[],
        fit_execution_authorized=False,
        test_outcomes_opened=False,
        selection_rule="same coverage-aware generated_checkpoint_criterion, frozen semantic subset and parent uncertainty; missing_label_fallback=stop; no TEST or per-arm endpoint rescue",
    )
    immutable(output / "prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare(args.campaign, args.output)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "source_commit", "steps", "fit_execution_authorized")}
        )
    )


if __name__ == "__main__":
    main()
