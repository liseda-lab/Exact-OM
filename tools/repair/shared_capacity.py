"""Prepare shared TRAIN scale and bounded paired engineering under concurrent DEV.

All weights are disposable. This does not select a model, admit a primary fit,
renew a prior qualification, or transmit annotations.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import time

from exact.repair.records import canonical_hash
from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read
from tools.repair.historical_regression import binding
from tools.repair.primary_runtime import FIT_GPU, DEV_GPU, owned_device, qualify
from tools.repair.shared_release import authenticate, bound, immutable

CONDITIONS = ("symbolic", "symbolic_plus_llm")


def scale_audit(cases, caches, labels, contexts, protocol):
    from exact.repair.semantic_fidelity import offline_plan_label
    from tools.repair.train import _protocol_profile

    profile = _protocol_profile(protocol)
    rows, receipts = [], {}
    for case in cases:
        if case.split != "train":
            continue
        records = labels.get(case.case_id, ())
        assignments = set()
        for packet, _ in records:
            for plan in (packet.plan_a, packet.plan_b):
                selection = dict(plan.complete_assignment)
                assignments.add(tuple(
                    next(i for i, c in enumerate(obj.candidates) if c.candidate_id == selection[obj.object_id])
                    for obj in case.problem.objects
                ))
        anchors = []
        symbolic = {l.assignment: l for l in caches[case.case_id].labels} if case.case_id in caches else {}
        for assignment in sorted(assignments):
            weak = offline_plan_label(
                case, assignment, profile, records,
                rating_aggregation=protocol["llm_labels"]["plan_rating_aggregation"],
                aggregate_receipts=receipts, rating_context_packets=contexts,
            )
            previous = symbolic.get(assignment)
            anchors.append(dict(
                assignment=assignment, weak_mask=weak is not None,
                weak_value=weak.benefit if weak else None,
                symbolic_value=previous.benefit if previous and previous.usable else None,
                fixed_cost=weak.cost if weak else None,
            ))
        rows.append(dict(case_id=case.case_id, family=case.family, control=case.control,
                         parent=case.structural_parent, symbolic_cache_available=case.case_id in caches,
                         comparisons=len(records), anchors=anchors))
    symbolic_values = [l.benefit for c in cases if c.split == "train" and c.case_id in caches
                       for l in caches[c.case_id].labels if l.usable]
    weak_values = [a["weak_value"] for r in rows for a in r["anchors"] if a["weak_mask"]]
    if not weak_values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in weak_values):
        raise ValueError("No finite unit-interval anchored TRAIN values")
    return dict(
        schema="exact-repair/train-only-weak-scale/v1", status="frozen",
        rule="identity anchored rubric scale; retain declared alpha=beta=0.2; no regression to the symbolic target",
        scale_alignment="TRAIN-anchored-rubric-identity/v1", alpha=0.2, beta=0.2,
        normalization="eligible_terms", symbolic_range=[min(symbolic_values), max(symbolic_values)],
        symbolic_usable_labels=len(symbolic_values), weak_range=[min(weak_values), max(weak_values)],
        weak_anchors=len(weak_values), rating_policy=protocol["llm_labels"]["plan_rating_aggregation"],
        representation_rule="exact native packet recovered with authenticated lossless round-trip proof",
        rows=rows, aggregates=receipts, fixed_costs=profile, test_opened=False,
        limitations=["Sparse generated TRAIN proxy, not equivalent symbolic and weak teachers",
                     "No real Conference semantic coverage", "No scale tuning on DEV or TEST outcomes"],
    )


def prepare(campaign, release_path, output, repository):
    from exact.repair.protocol import RepairProtocolV3, load_protocol_v3, training_projection_v3
    from exact.repair.semantic_fidelity import read_fidelity_training_artifact
    from tools.repair.common_training import load_release
    from tools.repair.corrective_campaign import source_identity

    campaign, output, repository = map(Path, (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    shared = read(release_path)
    labels = read_fidelity_training_artifact(authenticate(shared["labels"]), "train")
    contexts = bound(shared["rating_contexts"])
    common = bound(registry["common_training_preparation"]["preparation"])
    teacher = bound(registry["qualified_teacher_preparation"]["preparation"])
    cases, caches, _ = load_release(common["common_release"], common["audit_run"])
    protocol = training_projection_v3(load_protocol_v3(authenticate(teacher["protocols"][1])))
    audit = scale_audit(cases, caches, labels, contexts, protocol)
    immutable(output / "scale.json", audit)
    source = source_identity()
    if source["dirty_hash"] != __import__("hashlib").sha256(b"").hexdigest():
        raise ValueError("Commit tested source before freezing capacity")
    protocols, preparations = [], {}
    for ref in teacher["protocols"]:
        p = bound(ref)
        p["identity"].update(execution_authorized=False, code_hash=source["code_hash"],
                             dirty_hash=source["dirty_hash"], run_id=output.name + "-" + Path(ref["path"]).stem)
        p["losses"]["scale_alignment"] = audit["scale_alignment"] + ":" + binding(output / "scale.json")["sha256"]
        p["llm_labels"]["annotation_manifest"] = shared["labels"]["path"]
        path = output / "protocols" / Path(ref["path"]).name
        immutable(path, RepairProtocolV3.model_validate(p).model_dump(by_alias=True))
        protocols.append(binding(path))
        prep = dict(schema="exact-repair/common-training-preparation/v1",
                    release=common["common_release"], audit_run=common["audit_run"], protocol=binding(path))
        prep_path = output / "preparations" / path.name
        immutable(prep_path, dict(prep, hash=canonical_hash(prep)))
        preparations[path.stem] = binding(prep_path)
    manifest = dict(
        schema="exact-repair/shared-capacity/v1", purpose="disposable_engineering_only",
        shared_release=binding(release_path), labels=shared["labels"], rating_contexts=shared["rating_contexts"],
        scale=binding(output / "scale.json"), preparations=preparations,
        expected_train_cases=128, expected_dev_cases=32, seed=13,
        conditions=list(CONDITIONS), fit_seconds_per_condition=1050, dev_seconds_per_case=30,
        barrier_directory=str(output / "barrier"), deadline_epoch=1791997200.0,
        primary_fit_admitted=False, hosted_calls=0, heldout_outcomes_opened=False,
        missing_real_coverage=True, source_commit=source["revision"],
    )
    immutable(output / "manifest.json", manifest)
    jobs = []
    fit_id, dev_id = "qualify-shared-loss-5090-001", "qualify-shared-dev-2080-001"
    for lane, job_id, cpus, mem, gpu, gres, seconds in (
        ("fit", fit_id, 4, 24576, FIT_GPU, "gpu:rtx5090:1", 2800),
        ("development", dev_id, 5, 30720, DEV_GPU, "gpu:rtx2080ti:1", 2800),
    ):
        jobs.append(dict(
            id=job_id, stage="learning-engineering", budget_stages=["learning", "primary_training" if lane == "fit" else "secondary_learning"],
            seconds=seconds, slice_seconds=seconds, cleanup_seconds=15,
            deadline_epoch=1791997200.0, deadline_policy="defer", priority=220,
            resources=dict(cpus=cpus, memory_mb=mem, gpus=1, gres=gres), gpu_devices=[gpu],
            commands=[["{python}", "-m", "tools.repair.shared_capacity", lane, str(output / "manifest.json"), "{work}"]],
        ))
    ledger = read(campaign / "ledger.json")
    balances = {}
    for job in jobs:
        if any(s not in ledger["stages"] for s in job["budget_stages"]):
            raise ValueError("Engineering cannot start a fresh scientific stage clock")
        remaining = _stage_remaining(copy.deepcopy(ledger), job, time.time())
        if sum(j["seconds"] for j in jobs) > 0.7 * remaining:
            raise ValueError("Engineering reservation exceeds remaining 70 percent capacity")
        balances[job["id"]] = dict(remaining_seconds=remaining, reservation_seconds=job["seconds"])
    immutable(output / "admission.json", dict(
        status="engineering_slice_admitted", balances=balances, cpu_tokens=9, memory_mb=55296,
        primary_execution_authorized=False, counters_reset=False, prior_qualifications_replayed=False,
        worker_seconds_reserved=5600, actual_concurrency_requires_bound_overlap_receipts=True,
    ))
    spec = dict(
        repository=str(repository), campaign=str(campaign), allocation="14451",
        python="/home/pgcotovio/Exact-OM/.venv/bin/python", ledger=str(campaign / "ledger.json"),
        capacity=registry["capacity"], source_store=str(campaign / "sources"),
        protocol_source=protocols[0]["path"], jobs=jobs,
        input_files=[str(p) for p in output.rglob("*.json")] + [str(release_path), shared["labels"]["path"], shared["rating_contexts"]["path"]],
        purpose="Source-bound shared TRAIN weak scale, paired actual-loss/checkpoint and concurrent complete DEV engineering; no primary selection or paid calls",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(output / "batch-spec.json", campaign / "batches" / output.name)
    dependencies = ["annotate-controlled-train-00-001-byte-recovery-001", "annotate-controlled-train-01-001-byte-recovery-001"]
    descriptors = [prepare_dispatch(batch, j["id"], campaign / "attempts" / j["id"] / "001",
                                   tmux_socket=campaign / "supervisor/tmux.sock", depends_on=dependencies) for j in jobs]
    result = dict(schema="exact-repair/shared-capacity-preparation/v1", status="prepared_not_queued",
                  source_commit=source["revision"], shared_release=binding(release_path), scale=binding(output / "scale.json"),
                  manifest=binding(output / "manifest.json"), batch=binding(batch), descriptors=descriptors,
                  protocols=protocols, primary_fit_admitted=False, endpoint_frozen=False)
    immutable(output / "prepared.json", result)
    return result


def wait_for(path, deadline):
    while not path.exists():
        if time.time() + 15 >= deadline:
            raise TimeoutError("Concurrent engineering barrier exhausted owned deadline")
        time.sleep(min(0.5, max(0.01, deadline - time.time() - 15)))


def run_fit(manifest, output):
    output = Path(output)
    owned_device(FIT_GPU)
    deadline = min(float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]), manifest["deadline_epoch"])
    results = []
    for condition in CONDITIONS:
        target = output / condition
        started = Path(manifest["barrier_directory"]) / (condition + "-fit.json")
        if started.exists():
            raise ValueError("Interrupted engineering owner requires explicit recovery, not replay")
        immutable(started, dict(epoch=time.time(), step=os.environ["SLURM_STEP_ID"]))
        wait_for(started.with_name(condition + "-development.json"), deadline)
        os.environ["EXACT_REPAIR_PHASE_LOG"] = str(target / "phase-timings.jsonl")
        report = qualify(
            authenticate(manifest["preparations"][f"hgt-pair-{condition}-s13"]), target,
            seconds=min(manifest["fit_seconds_per_condition"], deadline-time.time()-20),
            shared_weak_labels=manifest["labels"], rating_contexts=manifest["rating_contexts"],
        )
        results.append(dict(condition=condition, report=binding(target / "report.json"), status=report["status"]))
    immutable(output / "report.json", dict(status="complete", results=results, primary_fit=False, manifest=manifest))


def run_development(manifest, output):
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.graph_schema import declared_metadata
    from exact.repair.model import RepairModel
    from exact.repair.protocol import load_protocol_v3, training_projection_v3
    from exact.repair.retrieval import retrieve_vocabulary
    from tools.repair import train
    from tools.repair.prepare import load_preparation
    from tools.repair.primary_runtime import scheduled_runtime_development

    output = Path(output)
    actual = owned_device(DEV_GPU)
    deadline = min(float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]), manifest["deadline_epoch"])
    prep = authenticate(manifest["preparations"]["hgt-pair-symbolic-s13"])
    protocol = training_projection_v3(load_protocol_v3(authenticate(read(prep)["protocol"])))
    cases, caches, _ = load_preparation(prep)
    config = train._protocol_arguments(protocol)
    config["profile"] = train._protocol_profile(protocol)
    development = scheduled_runtime_development(cases, caches, config)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(13)
    metadata = declared_metadata(config["graph_schema"])
    model = RepairModel(metadata, **{k: config[k] for k in (
        "graph_schema", "hidden_dim", "heads", "layers", "dropout", "encoder", "pairwise", "revision",
        "plan_risk", "support_enabled", "support_readout_identity", "pair_factor_bound")}).to("cuda")
    model.eval()
    preparation = EffectivePreparation(**{k: config[k] for k in (
        "max_graph_nodes", "max_graph_edges", "max_explanations", "max_text_tokens",
        "pair_factor_limit_per_object", "pair_max_pairs", "pair_max_factors", "retrieval_config", "revision")})
    options = {k: config[k] for k in (
        "selection_options", "case_cpu_seconds", "execution_schedule", "elementary_seconds", "circuit_limits",
        "compiler_cache_directory", "vtree_type", "candidate_cap", "mixtures", "profile", "pair_factor_limit_per_object",
        "pair_max_pairs", "pair_max_factors", "max_depth", "max_constructors", "max_circuit_nodes",
        "compile_seconds", "max_graph_nodes", "max_graph_edges", "max_explanations", "max_text_tokens",
        "quantization_scale", "retrieval_config")}
    options.update(draws_per_object=config["development_draws_per_object"], seed=13,
                   temperature=config["proposal_temperature"], proposal_arm="grammar_mixture")
    rows = []
    for condition in CONDITIONS:
        barrier = Path(manifest["barrier_directory"])
        wait_for(barrier / (condition + "-fit.json"), deadline)
        marker = barrier / (condition + "-development.json")
        if marker.exists():
            raise ValueError("Interrupted engineering DEV cannot renew row budgets")
        immutable(marker, dict(epoch=time.time(), step=os.environ["SLURM_STEP_ID"]))
        for case, cache in development:
            key = canonical_hash((condition, case.case_id))
            start = time.time()
            seconds = max(0, min(manifest["dev_seconds_per_case"], deadline-start-15))
            immutable(output / "starts" / (key + ".json"), dict(epoch=start, seconds=seconds))
            result = dict(status="not_attempted_deadline")
            if seconds > 2:
                retrieval = retrieve_vocabulary(case.problem, config=preparation.retrieval_config)
                graph = preparation.graph(case.problem, retrieval)
                result = train.generated_development(model, case, cache, seconds=max(0, seconds-(time.time()-start)),
                                                     graph=graph, **options)
            row = dict(case_id=case.case_id, family=case.family, control=case.control,
                       condition_load=condition, started_epoch=start, finished_epoch=time.time(),
                       elapsed_seconds=time.time()-start, result=result)
            immutable(output / "rows" / (key + ".json"), row)
            rows.append(binding(output / "rows" / (key + ".json")))
    immutable(output / "report.json", dict(status="complete", rows=rows, expected_rows=64,
              gpu_uuid=actual, peak_cuda_bytes=torch.cuda.max_memory_allocated(), hosted_calls=0,
              selected_model=False, scope="actual exposed DEV engineering at two predeclared matched load intervals",
              remaining_gate="Review complete timeout-inclusive rows and concurrency; freeze final DEV and common endpoint"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "fit", "development"))
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(*args.inputs)
    else:
        manifest, output = args.inputs
        (run_fit if args.mode == "fit" else run_development)(read(manifest), output)


if __name__ == "__main__":
    main()
