"""Frozen DEV-only full-pipeline engineering, using the production row executor.

No optimizer, hosted service, model selection or scientific evaluation admission.
Original engineering weights are converted losslessly; TEST remains closed.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read
from tools.repair.evaluation_ownership import PROFILE, lane_for, resources, validate_lane
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import SCHEMA, compile_rows, execute_rows
from tools.repair.primary_runtime import DEV_GPU
from tools.repair.shared_release import authenticate, bound, immutable, validate_completion

SCOPE = "representative_development"
MODEL = "engineering_hgt_pair_not_selected"


def fixed_cases(plan):
    declaration = bound(plan["development_declaration"])
    groups = {}
    for case in declaration["rows"]:
        if case["split"] != "development":
            raise ValueError("Capacity qualification requires DEV only")
        groups.setdefault((case["family"], case["control"]), []).append(case)
    chosen = [min(groups[key], key=lambda c: c["case_id"]) for key in sorted(groups)]
    if len(chosen) != 16 or len(plan["cases"]) != 16:
        raise ValueError("Expected the fixed sixteen family/control cases")
    for actual, frozen in zip(chosen, plan["cases"]):
        if any(actual.get(k) != v for k, v in frozen.items()):
            raise ValueError("Fixed DEV input identity changed")
    return plan["cases"]


def representative_rows(plan, template):
    """Reuse frozen operation/settings templates; never read TEST payloads."""
    compiled = compile_rows(template)
    case_id = next(
        r["case_id"] for r in compiled if r["cohort"] == "generated" and r["experiment"] == "E3"
    )
    prototypes = [
        r
        for r in compiled
        if r["case_id"] == case_id
        and (not r["adapter"]["model"] or r["adapter"]["model"] == "hgt-pair-symbolic_plus_llm-s13")
    ]
    rows = []
    for case in fixed_cases(plan):
        pool_id = None
        # Producer first; native deletion still never pays for rich generation.
        ordered = sorted(prototypes, key=lambda r: r["adapter"]["operation"] != "pool")
        for original in ordered:
            row = copy.deepcopy(original)
            for key in ("case_id", "structural_parent", "family", "control"):
                row[key] = case[key]
            row["input_identity"] = canonical_hash(case)
            row["protocol"] = plan["protocols"][0]
            row["engineering_template_row"] = original["id"]
            row["pool"] = pool_id if original["pool"] else None
            row["id"] = canonical_hash(
                {k: v for k, v in row.items() if k not in {"id", "status", "adapter"}}
            )
            row["adapter"].update(
                model=MODEL if original["adapter"]["model"] else None,
                pool_dependency=row["pool"],
                cost_identity=row["id"],
                output_directory="rows/" + row["id"],
            )
            if row["adapter"]["operation"] == "pool":
                pool_id = row["id"]
            rows.append(row)
    if {r["adapter"]["operation"] for r in rows} != set(plan["operations"]):
        raise ValueError("Representative operation coverage changed")
    return rows


def validate_manifest(manifest, *, check_visibility=True):
    if (
        manifest.get("scope") != SCOPE
        or manifest.get("schema") != SCHEMA
        or manifest.get("hosted_calls") != 0
        or manifest.get("optimizer_updates") != 0
        or manifest.get("test_payloads_allowed") is not False
    ):
        raise ValueError("Only closed-TEST engineering is admitted")
    schedule = bound(manifest["schedule"])
    plan = bound(schedule["fixed_plan"])
    cases = fixed_cases(plan)
    expected = representative_rows(plan, bound(schedule["template_schedule"]))
    if schedule["rows"] != expected:
        raise ValueError("Representative schedule differs from fixed templates")
    lane = manifest["ownership_lane"]
    if manifest["rows"] != [r for r in expected if lane_for(r) == lane]:
        raise ValueError("Engineering row denominator or settings changed")
    if manifest["cases"] != {c["case_id"]: c for c in cases}:
        raise ValueError("Engineering cannot read a different split")
    conversion = bound(manifest["engineering_model_conversion"])
    if (
        conversion.get("role") != MODEL
        or conversion.get("selected") is not False
        or conversion.get("optimizer_updates") != 0
        or manifest["models"] != {MODEL: conversion["artifact"]}
    ):
        raise ValueError("Engineering weights cannot be selected models")
    authenticate(conversion["artifact"])
    if manifest["public_inputs"] != schedule["public_inputs"]:
        raise ValueError("Engineering public input bindings changed")
    from tools.repair.primary_evaluation import _public_problem

    for case in cases:
        public = bound(manifest["public_inputs"][case["case_id"]])
        _public_problem(public, case)
    validate_lane(manifest, check_visibility=check_visibility)
    return schedule


def convert_engineering_weights(source, protocol_ref, output):
    """Serialization only: strict shape/schema checks, no forward or update."""
    import torch
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest

    torch.set_num_threads(1)
    state = torch.load(authenticate(source), weights_only=True, map_location="cpu")
    protocol = bound(protocol_ref)
    graph = protocol["model"]
    config = state["config"]
    expected = dict(
        encoder=graph["backbone"],
        hidden_dim=graph["hidden_width"],
        layers=graph["layers"],
        heads=graph["attention_heads"],
        graph_schema=graph["graph_schema"],
        revision="v3",
        pairwise=True,
        plan_risk=True,
        support_enabled=graph["support_enabled"],
    )
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Engineering checkpoint architecture differs from frozen protocol")
    model = RepairModel(state["metadata"], **config)
    model.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Engineering weights are nonfinite")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Never overwrite converted engineering weights")
    torch.save(
        dict(
            model_schema="exact-repair/model/v3",
            metadata=state["metadata"],
            config=config,
            state_dict=state["model"],
        ),
        output,
    )
    saved = torch.load(output, weights_only=True, map_location="cpu")
    if any(
        not torch.equal(value, saved["state_dict"][key]) for key, value in state["model"].items()
    ):
        raise ValueError("Engineering format conversion changed a tensor")
    return dict(
        role=MODEL,
        selected=False,
        source=source,
        protocol=protocol_ref,
        artifact=dict(**binding(output), model_hash=model_digest(model)),
        optimizer_updates=0,
        forward_passes=0,
        original_preserved=True,
        tensor_equality=True,
        purpose="pipeline engineering only; never a TEST model",
    )


def prepare(campaign, input_preparation, output, repository):
    from tools.repair.corrective_campaign import observable_queries
    from tools.repair.corrective_study import PUBLIC_INPUT_SCHEMA

    campaign, output, repository = map(Path, (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    prior = read(input_preparation)
    plan_ref = prior["representative_capacity_plan"]
    plan = bound(plan_ref)
    cases = fixed_cases(plan)
    rows = representative_rows(plan, bound(prior["schedule"]))
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise ValueError("Commit source before preparing immutable qualification")
    public_inputs = {}
    for case in cases:
        raw = bound(case["observable"])
        problem = read_record(raw)
        if problem.content_hash != case["input_hash"]:
            raise ValueError("Observable DEV input changed")
        scores = {
            o.object_id: dict(problem.evidence).get(o.object_id, {}).get("score")
            for o in problem.objects
        }
        available = all(o.kind == "mapping" for o in problem.objects) and all(
            type(v) in (int, float) and math.isfinite(v) for v in scores.values()
        )
        public = dict(
            schema=PUBLIC_INPUT_SCHEMA,
            metadata=case,
            input=raw,
            observable_queries=observable_queries(problem),
            **(dict(observable_mapping_scores=scores) if available else {}),
        )
        path = output / "public-inputs" / (canonical_hash(case) + ".json")
        immutable(path, public)
        public_inputs[case["case_id"]] = binding(path)
    original = next(r for r in registry["runs"] if r["id"] == "qualify-rtx2080ti")
    attempt = Path(original["completion_path"]).parent
    terminal = read(attempt / "completion.json")
    receipt = dict(
        completion=binding(attempt / "completion.json"),
        step=binding(attempt / "step.json"),
        outputs=binding(attempt / "outputs.json"),
        batch=binding(Path(terminal["batch"])),
        dispatch_nonce=original["dispatch_nonce"],
        step_id=original["step_id"],
        expected_status="complete",
    )
    completion, outputs, _, _ = validate_completion(receipt)
    source = dict(
        path=str(Path(completion["work"]) / "qualification-state.pt"),
        sha256=outputs["qualification-state.pt"],
    )
    conversion = convert_engineering_weights(
        source, plan["protocols"][0], output / "engineering.pt"
    )
    conversion["original_receipt"] = receipt
    immutable(output / "conversion.json", conversion)
    schedule = dict(
        schema="exact-repair/representative-capacity-schedule/v1",
        fixed_plan=plan_ref,
        template_schedule=prior["schedule"],
        rows=rows,
        public_inputs=public_inputs,
        cases=cases,
        source_commit=source_commit,
        scope=SCOPE,
        timeout_inclusive=True,
        hosted_calls=0,
        optimizer_updates=0,
        case_seconds=45,
        role=MODEL,
    )
    immutable(output / "schedule.json", schedule)
    # All queued primary DEV owners finish first; this also prevents the CPU
    # counterpart from spending its barrier budget while inference is waiting.
    primary = next(
        x
        for x in registry["pending_batches"]
        if x["id"] == "review-six-primary-pairs-and-freeze-001"
    )
    primary_dependencies = list(primary["depends_on"])
    ids = {
        lane: "qualify-representative-" + lane + "-001" for lane in ("pool", "native", "inference")
    }
    if any(r["id"] in ids.values() for r in registry["runs"] + registry["pending_batches"]):
        raise ValueError("Representative worker already registered; reconcile before preparing")
    manifests, jobs, dependencies = {}, [], {}
    nonces = {lane: uuid.uuid4().hex for lane in ids}
    for lane, job_id in ids.items():
        selected = [r for r in rows if lane_for(r) == lane]
        seconds = sum(r["case_seconds"] for r in selected) + (60 if lane == "pool" else 240)
        nonce = nonces[lane]
        manifest = dict(
            schema=SCHEMA,
            scope=SCOPE,
            execution_authorized=True,
            ownership_lane=lane,
            cpus=3,
            memory_mb=20000,
            device="cuda" if lane == "inference" else "cpu",
            gpu_uuid=DEV_GPU if lane == "inference" else None,
            concurrent_load_profile=PROFILE,
            dispatch_nonce=nonce,
            deadline_epoch=1791997200.0,
            ledger_path=str(campaign / "ledger.json"),
            source_commit=source_commit,
            schedule=binding(output / "schedule.json"),
            engineering_model_conversion=binding(output / "conversion.json"),
            cases={c["case_id"]: c for c in cases},
            public_inputs=public_inputs,
            models={MODEL: conversion["artifact"]},
            rows=selected,
            pool_producers={},
            hosted_calls=0,
            optimizer_updates=0,
            test_payloads_allowed=False,
        )
        if lane != "pool":
            manifest["concurrent_barrier"] = dict(
                directory=str(output / "barrier"),
                maximum_seconds=180,
                peers={
                    name: dict(
                        nonce=nonces[name], attempt=str(campaign / "attempts" / ids[name] / "001")
                    )
                    for name in ("native", "inference")
                },
            )
            producer = manifests["pool"]
            manifest["pool_producers"] = {
                r["pool"]: dict(
                    manifest=binding(output / "pool.json"),
                    dispatch_nonce=producer["dispatch_nonce"],
                    job_id=ids["pool"],
                    attempt=str(campaign / "attempts" / ids["pool"] / "001"),
                    work=str(campaign / "work" / ids["pool"]),
                )
                for r in selected
                if r["pool"]
            }
        immutable(output / (lane + ".json"), manifest)
        validate_manifest(manifest, check_visibility=False)
        manifests[lane] = manifest
        dependencies[lane] = (
            prior["next_review_dependencies"][:2]
            if lane == "pool"
            else list(dict.fromkeys([ids["pool"], *primary_dependencies]))
        )
        jobs.append(
            dict(
                id=job_id,
                stage="learning-engineering",
                budget_stages=["learning"]
                + (["secondary_learning"] if lane == "inference" else []),
                seconds=seconds,
                slice_seconds=seconds,
                cleanup_seconds=15,
                deadline_epoch=1791997200.0,
                deadline_policy="defer",
                priority=200,
                resources=resources(lane),
                gpu_devices=[DEV_GPU] if lane == "inference" else [],
                commands=[
                    [
                        "{python}",
                        "-m",
                        "tools.repair.representative_capacity",
                        "run",
                        str(output / (lane + ".json")),
                        "{work}",
                    ]
                ],
            )
        )
    ledger = read(campaign / "ledger.json")
    balances = {}
    total = sum(j["seconds"] for j in jobs)
    for job in jobs:
        if any(s not in ledger["stages"] for s in job["budget_stages"]):
            raise ValueError("Engineering cannot start a scientific stage clock")
        remaining = _stage_remaining(copy.deepcopy(ledger), job, time.time())
        if total > 0.7 * remaining:
            raise ValueError("Qualification exceeds 70 percent remaining capacity")
        balances[job["id"]] = dict(remaining_seconds=remaining, reservation_seconds=job["seconds"])
    immutable(
        output / "admission.json",
        dict(
            balances=balances,
            total_seconds=total,
            counters_reset=False,
            measured=False,
            stage_first_admissions_preserved=True,
            evaluation_stage_started=False,
            primary_dependencies=primary_dependencies,
        ),
    )
    spec = dict(
        repository=str(repository),
        campaign=str(campaign),
        allocation="14451",
        python="/home/pgcotovio/Exact-OM/.venv/bin/python",
        ledger=str(campaign / "ledger.json"),
        capacity=registry["capacity"],
        source_store=str(campaign / "sources"),
        protocol_source=plan["protocols"][0]["path"],
        jobs=jobs,
        input_files=[str(p) for p in output.rglob("*") if p.is_file()]
        + [plan_ref["path"], prior["schedule"]["path"], source["path"]]
        + [c["observable"]["path"] for c in cases]
        + [p["path"] for p in plan["protocols"]],
        purpose="Fixed sixteen DEV full-pipeline engineering; no TEST, optimization or hosted calls",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(output / "batch-spec.json", campaign / "batches" / output.name)
    descriptors = [
        prepare_dispatch(
            batch,
            ids[lane],
            campaign / "attempts" / ids[lane] / "001",
            tmux_socket=campaign / "supervisor/tmux.sock",
            nonce=manifests[lane]["dispatch_nonce"],
            depends_on=dependencies[lane],
        )
        for lane in ids
    ]
    result = dict(
        schema="exact-repair/representative-capacity-preparation/v1",
        status="prepared_not_queued",
        input_preparation=binding(Path(input_preparation)),
        fixed_plan=plan_ref,
        source_commit=source_commit,
        batch=binding(batch),
        descriptors=descriptors,
        schedule=binding(output / "schedule.json"),
        conversion=binding(output / "conversion.json"),
        admission=binding(output / "admission.json"),
        manifests={lane: binding(output / (lane + ".json")) for lane in ids},
        primary_dependencies=primary_dependencies,
        measured=False,
        steps=[],
    )
    immutable(output / "prepared.json", result)
    audit_id = "audit-representative-capacity-001"
    audit_job = dict(
        id=audit_id,
        stage="learning-engineering",
        budget_stages=["learning"],
        seconds=360,
        slice_seconds=360,
        cleanup_seconds=15,
        deadline_epoch=1791997200.0,
        deadline_policy="defer",
        priority=200,
        resources=dict(cpus=1, gpus=0, memory_mb=8192, gres="none"),
        gpu_devices=[],
        commands=[
            [
                "{python}",
                "-m",
                "tools.repair.representative_capacity",
                "audit",
                str(output / "prepared.json"),
                "{work}",
            ]
        ],
    )
    if total + audit_job["seconds"] > 0.7 * _stage_remaining(
        copy.deepcopy(ledger), audit_job, time.time()
    ):
        raise ValueError("Terminal audit exceeds remaining learning capacity")
    audit_spec = dict(
        spec,
        jobs=[audit_job],
        input_files=[str(output / "prepared.json")],
        purpose="Authenticate representative pipeline outputs, overlap and usage",
    )
    immutable(output / "audit-batch-spec.json", audit_spec)
    audit_batch = freeze(
        output / "audit-batch-spec.json", campaign / "batches" / (output.name + "-audit")
    )
    audit_descriptor = prepare_dispatch(
        audit_batch,
        audit_id,
        campaign / "attempts" / audit_id / "001",
        tmux_socket=campaign / "supervisor/tmux.sock",
        depends_on=list(ids.values()),
    )
    dispatch = dict(
        preparation=binding(output / "prepared.json"),
        audit_batch=binding(audit_batch),
        descriptors=descriptors + [audit_descriptor],
        qualification_run_ids=list(ids.values()),
        audit_run_id=audit_id,
        status="prepared_not_queued",
        steps=[],
        source_commit=source_commit,
    )
    immutable(output / "dispatch-prepared.json", dispatch)
    return dispatch


def wait_for_peer(manifest):
    """Finite charged barrier; a missing peer is an engineering failure, not data."""
    from tools.repair.evaluation_ownership import validate_owner

    validate_manifest(manifest)
    validate_owner(manifest, read(manifest["ledger_path"]))
    barrier = manifest.get("concurrent_barrier")
    if not barrier:
        return
    lane = manifest["ownership_lane"]
    directory = Path(barrier["directory"])
    identity = dict(schedule=manifest["schedule"], source_commit=manifest["source_commit"])
    ready = dict(
        **identity,
        lane=lane,
        dispatch_nonce=manifest["dispatch_nonce"],
        step_id=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
    )
    immutable(directory / (lane + ".json"), ready)
    limit = min(
        time.time() + barrier["maximum_seconds"],
        float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]) - 30,
    )
    while time.time() < limit:
        peers = [directory / (p + ".json") for p in barrier["peers"]]
        if all(p.exists() for p in peers):
            if any(any(read(p).get(k) != v for k, v in identity.items()) for p in peers):
                raise ValueError("Concurrent peer belongs to another schedule")
            ledger = read(manifest["ledger_path"])
            for name, expected in barrier["peers"].items():
                peer = read(directory / (name + ".json"))
                step = read(Path(expected["attempt"]) / "step.json")
                owner = ledger["attempts"].get(expected["attempt"], {})
                if (
                    peer["dispatch_nonce"] != expected["nonce"]
                    or step.get("dispatch_nonce") != expected["nonce"]
                    or step.get("step_id") != peer["step_id"]
                    or owner.get("status") == "settled"
                    or not owner
                ):
                    raise ValueError("Concurrent peer lacks a live nonce-bound charged owner")
            return
        time.sleep(min(0.25, max(0, limit - time.time())))
    raise TimeoutError("Concurrent peer did not arrive within charged barrier reservation")


def run(manifest, output):
    wait_for_peer(manifest)
    return execute_rows(manifest, output)


def recovery_rows(manifest):
    """Authenticate and retain one failed cleanup prefix without replay.

    The failed case's deadline has expired. Process retirement permits the
    remaining unattempted cases to run, never another attempt at that case.
    """
    reference = manifest.get("technical_recovery")
    if not reference:
        return {}
    recovery = bound(reference)
    prior = bound(recovery["manifest"])
    if prior.get("technical_recovery"):
        raise ValueError("A further recovery requires a separate same-cause review")
    changed = {"dispatch_nonce", "source_commit", "concurrent_barrier", "technical_recovery"}
    if {k: v for k, v in prior.items() if k not in changed} != {
        k: v for k, v in manifest.items() if k not in changed
    }:
        raise ValueError("Technical recovery changed scientific rows or limits")
    completion, outputs, batch, job = validate_completion(recovery["receipt"])
    if completion["status"] != "failed" or recovery["manifest"]["path"] not in job["commands"][0]:
        raise ValueError("Technical recovery requires its original failed worker")
    retirement = bound(recovery["retirement"])
    if (
        retirement.get("retired_step") != completion["step_id"]
        or completion["step_id"] in retirement["live_steps"]
        or retirement["observed_epoch"] < completion["finished_epoch"]
        or retirement.get("all_owned_processes_retired") is not True
    ):
        raise ValueError("Failed worker cleanup has not been reconciled")
    work = Path(completion["work"])
    for relative, digest in outputs.items():
        authenticate(dict(path=str(work / relative), sha256=digest))
    report = read(work / "report.json")
    if outputs.get("report.json") != binding(work / "report.json")["sha256"] or (
        report["manifest_identity"] != canonical_hash(prior)
        or [r["id"] for r in report["rows"]] != [r["id"] for r in prior["rows"]]
    ):
        raise ValueError("Failed report identity or denominator differs")
    attempted = [r for r in report["rows"] if r["status"] != "not_attempted"]
    if len(attempted) != 1:
        raise ValueError("Recovery must preserve exactly the diagnosed failed prefix")
    row = attempted[0]
    frozen = next(r for r in prior["rows"] if r["id"] == row["id"])
    saved_path = work / "rows" / row["id"] / "completion.json"
    saved = read(saved_path)
    if (
        row["id"] != recovery["failed_row_id"]
        or saved != {k: v for k, v in row.items() if k != "id"}
        or outputs.get(str(saved_path.relative_to(work))) != binding(saved_path)["sha256"]
        or saved["identity"] != canonical_hash((canonical_hash(prior), frozen))
        or saved["status"] != "cleanup_incomplete"
        or saved.get("cleanup_complete") is not False
        or saved.get("detail") != "nested worker cleanup incomplete"
        or not 0 < saved["charged_seconds"] < frozen["case_seconds"]
    ):
        raise ValueError("Recovery is not the diagnosed technical cleanup failure")
    return {
        row["id"]: dict(
            identity=canonical_hash((canonical_hash(manifest), frozen)),
            status="prior_implementation_failure",
            result=None,
            cleanup_complete=True,
            prior_cleanup_complete=False,
            charged_seconds=saved["charged_seconds"],
            prior_charged_seconds=saved["charged_seconds"],
            artifacts=[binding(saved_path), recovery["retirement"]],
            prior_attempt=recovery["receipt"],
            shared_cost_reference=saved.get("shared_cost_reference"),
            cost_rule="Prior failed row charged once; expired case never replayed; descendant retirement authenticated separately",
        )
    }


def audit(prepared, output):
    """Authenticate terminal workers and report measured coverage without replay."""
    prepared = read(prepared)
    reports, terminals, receipts, operations = {}, {}, {}, []
    for descriptor in prepared["descriptors"]:
        attempt = Path(descriptor["launch"]["step_path"]).parent
        terminal = read(attempt / "completion.json")
        receipt = dict(
            completion=binding(attempt / "completion.json"),
            step=binding(attempt / "step.json"),
            outputs=binding(attempt / "outputs.json"),
            batch=prepared.get("worker_batches", {}).get(descriptor["id"], prepared["batch"]),
            dispatch_nonce=descriptor["launch"]["nonce"],
            step_id=terminal["step_id"],
            expected_status="complete",
        )
        completion, outputs, _, job = validate_completion(receipt)
        manifest = bound(
            next(ref for ref in prepared["manifests"].values() if ref["path"] in job["commands"][0])
        )
        lane = manifest["ownership_lane"]
        validate_manifest(manifest, check_visibility=False)
        work = Path(completion["work"])
        for relative, digest in outputs.items():
            authenticate(dict(path=str(work / relative), sha256=digest))
        report = read(work / "report.json")
        if report["manifest_identity"] != canonical_hash(manifest) or [
            r["id"] for r in report["rows"]
        ] != [r["id"] for r in manifest["rows"]]:
            raise ValueError("Capacity output denominator/identity differs")
        from tools.repair.primary_evaluation import validate_saved

        for row, reported in zip(manifest["rows"], report["rows"]):
            if reported["status"] == "not_attempted":
                continue
            saved = read(work / "rows" / row["id"] / "completion.json")
            if saved != {k: v for k, v in reported.items() if k != "id"}:
                raise ValueError("Capacity row and terminal report differ")
            validate_saved(saved, canonical_hash((canonical_hash(manifest), row)))
            operation = work / "rows" / row["id"] / "payload" / "operation.json"
            if operation.exists():
                operations.append(
                    dict(
                        **read(operation),
                        terminal_status=saved["status"],
                        result_status=(saved.get("result") or {}).get("status"),
                    )
                )
        reports[lane], terminals[lane], receipts[lane] = report, completion, receipt
    intervals = {
        lane: [
            (r["started_epoch"], r["finished_epoch"])
            for r in report["rows"]
            if "started_epoch" in r
        ]
        for lane, report in reports.items()
    }
    overlap = sum(
        max(0, min(a1, b1) - max(a0, b0))
        for a0, a1 in intervals["native"]
        for b0, b1 in intervals["inference"]
    )
    incomplete = [
        r["id"]
        for report in reports.values()
        for r in report["rows"]
        if r["status"]
        in {"error", "interrupted_unknown", "not_attempted", "prior_implementation_failure"}
        or not r.get("cleanup_complete")
    ]
    result = dict(
        schema="exact-repair/representative-capacity-review/v1",
        status="complete",
        receipts=receipts,
        row_reports=reports,
        concurrent_load_profile=PROFILE,
        source_commit=prepared["source_commit"],
        active_native_inference_overlap_seconds=overlap,
        incomplete_rows=incomplete,
        capacity_measured=all(intervals.values()),
        lifecycle_qualified=not incomplete,
        operation_entries=operations,
        concurrent_work_observed=overlap > 0,
        test_execution_admitted=False,
        remaining_stage_projection_passed=False,
        worker_usage={
            k: {
                field: v[field]
                for field in ("elapsed_seconds", "cpu_seconds", "peak_rss_mb", "step_id")
            }
            for k, v in terminals.items()
        },
        qualification_rule="No performance threshold; terminal unknown/unsupported/timeouts retained. Review actual operation reachability and live remaining capacity before TEST admission.",
        hosted_calls=0,
        optimizer_updates=0,
        test_outcomes_opened=False,
        technical_recovery=prepared.get("technical_recovery"),
    )
    immutable(Path(output) / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.campaign, args.input, args.output, args.repository)
    elif args.mode == "run":
        run(read(args.input), args.output)
    else:
        audit(args.input, args.output)


if __name__ == "__main__":
    main()
