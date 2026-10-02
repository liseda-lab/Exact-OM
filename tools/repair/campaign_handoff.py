"""Small adapters for frozen repair batches and the existing deterministic dispatcher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.protocol import load_protocol_v3, training_projection_v3
from exact.repair.records import canonical_hash
from tools.repair.prepare import load_preparation, save_preparation


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_dependencies(protocol):
    """Only the declared architecture contrast and cache location may differ.

    Input, policy, query/target, source, preparation, collection and cost settings
    remain identical. The original cache records are never rewritten.
    """
    value = copy.deepcopy(protocol)
    value.pop("resolved_v3_protocol_hash", None)
    value["identity"].pop("run_id")
    value["circuit"].pop("cache_directory")
    for section in ("model", "graph"):
        for key in ("backbone", "pair_benefit", "encoder"):
            value[section].pop(key, None)
    return canonical_hash(value)


def reuse_preparation(source, protocol_path, output):
    """Admit unchanged teacher labels into an explicitly different model arm."""
    target = training_projection_v3(load_protocol_v3(Path(protocol_path), for_execution=True))
    cases, caches, report = load_preparation(Path(source))
    original = report["protocol"]
    if report["protocol_hash"] != canonical_hash(original):
        raise ValueError("Source preparation protocol identity is inconsistent")
    if label_dependencies(original) != label_dependencies(target):
        raise ValueError("Label dependencies changed; cached supervision cannot be reused")
    if any(case.split == "test" and case.case_id in caches for case in cases):
        raise ValueError("Training preparation must not contain test supervision")
    provenance = {
        "schema": "exact-repair/architecture-label-reuse/v1",
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "source_protocol_hash": report["protocol_hash"],
        "target_protocol_hash": canonical_hash(target),
        "label_dependency_hash": label_dependencies(target),
        "cache_hashes": {key: canonical_hash(cache) for key, cache in caches.items()},
        "original_label_seconds": report.get("label_seconds", 0),
        "original_label_cpu_seconds": report.get("label_cpu_seconds"),
        "labels_recomputed": False,
    }
    amended = {
        **report,
        "protocol": target,
        "protocol_hash": canonical_hash(target),
        "reuse_provenance": provenance,
    }
    output = Path(output)
    if output.exists():
        previous_cases, previous_caches, previous_report = load_preparation(output)
        if (previous_cases, previous_caches, previous_report) != (cases, caches, amended):
            raise ValueError("Existing derived preparation differs; preserve and inspect it")
    else:
        save_preparation(output, cases, amended, caches)
    return provenance


def preparation_gate(source, output):
    cases, caches, report = load_preparation(Path(source))
    selected = [c for c in cases if c.split in {"train", "development"}]
    incomplete = [
        c.case_id for c in selected if c.case_id not in caches or not caches[c.case_id].complete
    ]
    heldout_labels = [c.case_id for c in cases if c.split == "test" and c.case_id in caches]
    evidence = {
        "schema": "exact-repair/preparation-gate/v1",
        "source_sha256": file_hash(source),
        "status": (
            "complete" if not incomplete and not heldout_labels and selected else "incomplete"
        ),
        "case_counts": dict(Counter(c.split for c in cases)),
        "parent_counts": dict(
            Counter(split for _, split in {(c.structural_parent, c.split) for c in cases})
        ),
        "scheduled_label_cases": len(selected),
        "complete_label_cases": sum(
            c.case_id in caches and caches[c.case_id].complete for c in selected
        ),
        "incomplete_cases": incomplete,
        "heldout_labels": heldout_labels,
        "label_seconds": report.get("label_seconds", 0),
        "scope": "Small generated development cohort; no comparative-quality claim",
    }
    write_artifact(output, evidence)
    if evidence["status"] != "complete":
        raise ValueError("Preparation gate incomplete; inspect its recorded coverage")
    return evidence


def qualify_gpu(prepared, output):
    import gc
    import time

    import pyowl_core as owl
    import torch

    from exact.repair.graph import FEATURE_SCHEMA_V3, build_observable_graph
    from exact.repair.model import RepairModel

    cases, _, _ = load_preparation(Path(prepared))
    case = next(c for c in cases if c.split == "train")
    symbols = set()
    for obj in case.problem.objects:
        for candidate in obj.candidates:
            for axiom in (*candidate.axioms, *candidate.active_expressions):
                symbols.update(owl.signature(axiom))
    graph = build_observable_graph(
        case.problem.objects,
        fixed_axioms=case.problem.fixed_axioms,
        source_axioms=case.problem.source_axioms,
        target_axioms=case.problem.target_axioms,
        retrieved_symbols=symbols,
        feature_schema=FEATURE_SCHEMA_V3,
    )
    if torch.cuda.device_count() != 1:
        raise ValueError("This qualification requires exactly one allocated CUDA device")
    torch.set_num_threads(1)
    rows = []
    for encoder in ("hgt", "rgcn", "none"):
        torch.manual_seed(13)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        model = RepairModel(
            graph.metadata,
            encoder=encoder,
            hidden_dim=128,
            heads=4,
            layers=3,
            pairwise=True,
            revision="v3",
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        memory = model.encode(graph)
        unary, pairs = model.score_inventory(
            case.problem.objects, memory, interaction_pairs=((0, 1),)
        )
        assignment = tuple(0 for _ in case.problem.objects)
        risk = model.plan_risk_logit(case.problem.objects, memory, assignment)
        loss = (
            sum(v.square().mean() for v in unary)
            + sum(v.square() for v in pairs.values())
            + risk.square()
        )
        loss.backward()
        if not torch.isfinite(loss).item() or not all(
            torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None
        ):
            raise ValueError("Nonfinite model loss or gradient")
        optimizer.step()
        torch.cuda.synchronize()
        rows.append(
            dict(
                encoder=encoder,
                seconds=time.monotonic() - started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                graph_nodes=len(graph.nodes),
                graph_edges=len(graph.edges),
            )
        )
        del model, optimizer, memory, unary, pairs, risk, loss
        gc.collect()
    evidence = dict(
        status="complete",
        scope="component qualification, not full-batch capacity",
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        rows=rows,
    )
    write_artifact(output, evidence)
    return evidence


def dispatched_worker(batch_path, job_id, attempt, nonce):
    """Bind batch receipts to the existing dispatcher's nonce; no monitoring loop."""
    from tools.repair.batch import run

    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=True)
    identity = dict(
        step_id=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"], dispatch_nonce=nonce
    )
    write_artifact(attempt / "step.json", identity)
    code = run(batch_path, job_id, attempt)
    for name in ("status.json", "completion.json"):
        path = attempt / name
        write_artifact(path, {**json.loads(path.read_text()), **identity})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reuse = commands.add_parser("reuse")
    reuse.add_argument("source")
    reuse.add_argument("protocol")
    reuse.add_argument("output")
    for name in ("gate", "qualify-gpu"):
        command = commands.add_parser(name)
        command.add_argument("source")
        command.add_argument("output")
    worker = commands.add_parser("worker")
    worker.add_argument("batch")
    worker.add_argument("job")
    worker.add_argument("attempt")
    worker.add_argument("nonce")
    args = parser.parse_args()
    if args.command == "worker":
        return dispatched_worker(args.batch, args.job, args.attempt, args.nonce)
    if args.command == "reuse":
        result = reuse_preparation(args.source, args.protocol, args.output)
    elif args.command == "gate":
        result = preparation_gate(args.source, args.output)
    else:
        result = qualify_gpu(args.source, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
