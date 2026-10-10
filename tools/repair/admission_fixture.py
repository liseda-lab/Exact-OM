"""Finite concurrent component probe; authored TRAIN fixtures, never TEST admission."""

from __future__ import annotations

import argparse
import dataclasses
import os
import resource
import subprocess
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.evaluation_ownership import PROFILE, resources
from tools.repair.historical_regression import binding
from tools.repair.primary_runtime import DEV_GPU, owned_device
from tools.repair.shared_release import immutable


def validate_owner(manifest, lane, ledger, step_id):
    expected = resources(lane)
    owners = [
        (Path(key), row)
        for key, row in ledger["attempts"].items()
        if row["status"] != "settled"
        and (Path(key) / "step.json").exists()
        and read(Path(key) / "step.json").get("step_id") == step_id
    ]
    if len(owners) != 1:
        raise ValueError("Fixture requires one charged owner")
    path, owner = owners[0]
    stages = {"learning", "secondary_learning"} if lane == "inference" else {"learning"}
    if (
        any(owner["resources"][key] != value for key, value in expected.items())
        or set(owner["budget_stages"]) != stages
        or owner["gpu_devices"] != ([DEV_GPU] if lane == "inference" else [])
        or read(path / "step.json")["dispatch_nonce"] != manifest["nonces"][lane]
    ):
        raise ValueError("Fixture owner resource/stage/nonce mismatch")


def native(case):
    from exact.repair.owl import OwlVerifier

    problem = case.problem
    selected = tuple(obj.candidates[-1] for obj in problem.objects)
    return (
        OwlVerifier("auto", backend="auto")
        .check_assignment(
            problem.fixed_axioms,
            selected,
            problem.policy.monitored_classes,
            required=problem.policy.required,
            prohibited=problem.policy.prohibited,
            feasibility_only=True,
        )
        .to_dict()
    )


def remaining(deadline, case_deadline, cleanup=2):
    return max(0, min(deadline, case_deadline) - time.time() - cleanup)


def run(manifest_path, lane, output):
    import torch
    from exact.repair.graph import EffectivePreparation, GraphNode, ObservableGraph
    from exact.repair.graph_schema import declared_metadata, generic_graph_schema
    from exact.repair.model import RepairModel, repair_benefits
    from exact.repair.retrieval import retrieve_vocabulary
    from exact.repair.workers import bounded_call
    from tools.repair.corpus import generate_corpus

    manifest = read(manifest_path)
    if manifest["profile"] != PROFILE or manifest["scope"] != "authored_train_components":
        raise ValueError("Fixture scope/profile changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "started.json").exists():
        raise ValueError("Terminal or interrupted component probe cannot be replayed")
    deadline = min(float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]), manifest["deadline_epoch"])
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    validate_owner(manifest, lane, read(os.environ["EXACT_REPAIR_STAGE_LEDGER"]), step)
    gpu = owned_device(DEV_GPU) if lane == "inference" else None
    if lane == "native" and torch.cuda.device_count():
        raise ValueError("Native fixture must have no visible GPU")
    device_processes = []
    if lane == "inference":
        listing = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=min(5, max(0.1, remaining(deadline, deadline))),
        )
        for line in listing.splitlines():
            uuid, pid = (item.strip() for item in line.split(","))
            if uuid.lower() == DEV_GPU.lower():
                device_processes.append(int(pid))
        if set(device_processes) - {os.getpid()}:
            raise ValueError("Allocation conflict: foreign process on the admitted GPU UUID")
    immutable(
        output / "started.json", dict(step=step, nonce=manifest["nonces"][lane], epoch=time.time())
    )
    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(manifest["seed"])
    cases = generate_corpus(
        split_counts={"train": 1, "development": 0, "test": 0},
        siblings_per_parent=1,
        families=("range", "interaction_complementary"),
        coherent_controls=True,
        revision="v3",
    )
    assert cases and all(case.split == "train" for case in cases)
    model, graphs, pairs = None, {}, {}
    schema = generic_graph_schema()
    metadata = declared_metadata(schema)
    if len(metadata[0]) != 13 or len(metadata[1]) != 553:
        raise ValueError("Installed full-schema declaration changed")
    if lane == "inference":
        preparation = EffectivePreparation(revision="v3")
        typed = tuple(
            GraphNode("schema:" + kind, kind, (("observed", 1.0),)) for kind in metadata[0]
        )
        edges = tuple(("schema:" + a, r, "schema:" + b) for a, r, b in metadata[1])
        for case in cases:
            base = preparation.graph(case.problem, retrieve_vocabulary(case.problem))
            graph = ObservableGraph(
                base.nodes + typed,
                base.edges + edges,
                base.object_nodes,
                feature_schema=base.feature_schema,
            )
            graphs[case.case_id] = graph
            pairs[case.case_id] = preparation.pairs(case.problem, graph).pairs
        model = (
            RepairModel(
                metadata,
                graph_schema=schema,
                encoder="hgt",
                hidden_dim=128,
                heads=4,
                layers=3,
                dropout=0.1,
                revision="v3",
                pairwise=True,
            )
            .to("cuda")
            .eval()
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    shared = Path(manifest["barrier_directory"])
    ready = dict(step=step, nonce=manifest["nonces"][lane], epoch=time.time())
    immutable(shared / (lane + ".json"), ready)
    other = "native" if lane == "inference" else "inference"
    barrier_deadline = min(deadline - 10, started + manifest["barrier_seconds"])
    peer = shared / (other + ".json")
    while not peer.exists() and time.time() < barrier_deadline:
        time.sleep(min(0.1, max(0, barrier_deadline - time.time())))
    peer_ready = read(peer) if peer.exists() else None
    if peer_ready and peer_ready["nonce"] != manifest["nonces"][other]:
        raise ValueError("Foreign fixture barrier nonce")
    rows = []
    for index in range(manifest["rows"][lane]):
        case = cases[index % len(cases)]
        row = dict(
            index=index,
            case_id=case.case_id,
            input_hash=case.problem.content_hash,
            family=case.family,
            control=case.control,
            status="not_attempted_deadline",
        )
        case_deadline = min(deadline - 5, time.time() + manifest["case_seconds"])
        seconds = remaining(deadline - 5, case_deadline)
        if peer_ready is None:
            row["status"] = "not_attempted_peer_unavailable"
        elif seconds > 0:
            begin = time.time()
            cpu_begin = time.process_time()
            if lane == "native":
                outcome = bounded_call(
                    native, case, timeout=seconds, memory_mb=18000, cpu_seconds=min(3 * seconds, 60)
                )
                row.update(dataclasses.asdict(outcome))
                if not outcome.cleanup_complete:
                    raise RuntimeError("Fixture native descendants not cleaned up")
            else:
                with torch.no_grad():
                    memory = model.encode(graphs[case.case_id])
                    unary, factors = model.score_inventory(
                        case.problem.objects, memory, interaction_pairs=pairs[case.case_id]
                    )
                    assignment = tuple(len(obj.candidates) - 1 for obj in case.problem.objects)
                    values = repair_benefits([assignment], unary, factors)
                    risk = model.plan_risk_logit(case.problem.objects, memory, assignment)
                    torch.cuda.synchronize()
                row.update(
                    status=(
                        "complete"
                        if bool(torch.isfinite(values).all() and torch.isfinite(risk).all())
                        else "nonfinite"
                    )
                )
            row.update(
                started_epoch=begin,
                finished_epoch=time.time(),
                elapsed_seconds=time.time() - begin,
                coordinator_cpu_seconds=time.process_time() - cpu_begin,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            if lane == "inference" and time.time() > case_deadline:
                row["status"] = "resource_limited_case_deadline"
        rows.append(row)
        write_artifact(
            output / "progress.json", dict(rows=rows, expected_rows=manifest["rows"][lane])
        )
    result = dict(
        schema="exact-repair/admission-component-fixture/v1",
        status=(
            "complete" if all(r["status"] == "complete" for r in rows) else "incomplete_component"
        ),
        manifest=binding(Path(manifest_path)),
        lane=lane,
        profile=PROFILE,
        step_id=step,
        dispatch_nonce=manifest["nonces"][lane],
        gpu_uuid=gpu,
        gpu_processes_before_setup=device_processes,
        resources=resources(lane),
        started_epoch=started,
        finished_epoch=time.time(),
        rows=rows,
        peer_ready=peer_ready,
        elapsed_seconds=time.time() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if lane == "inference" else 0,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if lane == "inference" else 0,
        graph_schema_hash=canonical_hash(schema),
        node_types=13,
        relations=553,
        layers=3,
        width=128,
        heads=4,
        actual_model_config=model.config if model is not None else None,
        actual_parameter_count=(
            sum(p.numel() for p in model.parameters()) if model is not None else 0
        ),
        optimizer_updates=0,
        selected_weights=False,
        test_payloads_opened=False,
        hosted_calls=0,
        capacity_admitted=False,
        limitation="component fixtures only; actual-case and combined-loss admission remain required",
    )
    immutable(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("lane", choices=("native", "inference"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.lane, args.output)


if __name__ == "__main__":
    main()
