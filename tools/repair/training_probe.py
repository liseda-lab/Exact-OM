"""Observable-only full-schema capacity and proposal-gradient qualification.

Fresh disposable weights; no optimizer, teacher, labels, decoder or checkpoint
selection. Every development row and bounded failure remains in the report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair.historical_regression import binding, verify_binding

ARMS = tuple(
    f"{encoder}-{head}" for encoder in ("hgt", "rgcn", "none") for head in ("unary", "pairwise")
)


def schedule(plan):
    if (
        plan.get("schema") != "exact-repair/training-probe-plan/v1"
        or plan.get("split") != "development"
        or plan.get("expected_cases") != 32
        or plan.get("heldout_use") is not False
        or plan.get("model_fitting") is not False
        or plan.get("arms") != list(ARMS)
    ):
        raise ValueError("Probe requires the complete development-only six-arm boundary")
    release = verify_binding(plan["release"])
    rows = release["rows"]
    if len(rows) != 32 or len({r["case_id"] for r in rows}) != 32:
        raise ValueError("Development probe denominator changed")
    if any(r["split"] != "development" or r["status"] != "materialized" for r in rows):
        raise ValueError("Development probe split/status changed")
    # Never deserialize evaluator records. Bind only observable rows.
    return [
        {
            k: r[k]
            for k in (
                "case_id",
                "structural_parent",
                "family",
                "control",
                "input_hash",
                "observable",
            )
        }
        for r in rows
    ]


def gradient_summary(model, loss, *, retain_graph=False):
    import torch

    names, parameters = zip(*model.named_parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    present = [(name, grad) for name, grad in zip(names, gradients) if grad is not None]
    if (
        not torch.isfinite(loss).item()
        or not present
        or any(not torch.isfinite(grad).all().item() for _, grad in present)
    ):
        raise ValueError("Nonfinite or missing diagnostic gradients")
    nonzero = [name for name, grad in present if torch.count_nonzero(grad).item()]
    if not nonzero:
        raise ValueError("All diagnostic gradients are zero")
    return dict(
        loss=float(loss.detach()),
        parameters_with_gradients=len(present),
        nonzero_parameters=nonzero,
    )


def probe(record, settings, declaration, arm, directory):
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.graph_schema import training_metadata
    from exact.repair.grammar import mapping_grammar, with_immutable_context
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import proposal_distribution
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(settings["seed"])
    device = settings["device"]
    if device == "cuda":
        if (
            torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != settings["device_name"]
        ):
            raise ValueError("Unexpected allocated GPU")
        torch.cuda.manual_seed_all(settings["seed"])
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    problem = read_record(record)
    preparation = EffectivePreparation(
        **settings["graph"], retrieval_config=RetrievalConfig(**settings["retrieval"])
    )
    retrieval = retrieve_vocabulary(problem, config=preparation.retrieval_config)
    graph = preparation.graph(problem, retrieval)
    encoder, head = arm.split("-")
    pairs = preparation.pairs(problem, graph, enabled=head == "pairwise")
    model = RepairModel(
        training_metadata(graph.metadata, declaration),
        graph_schema=declaration,
        encoder=encoder,
        pairwise=head == "pairwise",
        **settings["model"],
    ).to(device)
    memory = model.encode(graph)
    unary, factors = model.score_inventory(problem.objects, memory, interaction_pairs=pairs.pairs)
    risk = model.plan_risk_logit(
        problem.objects, memory, tuple(0 for _ in problem.objects), supports=graph.admitted_supports
    )
    diagnostic = torch.stack(
        [v.square().mean() for v in unary]
        + [v.square() for v in factors.values()]
        + [risk.square()]
    ).mean()
    readouts = gradient_summary(model, diagnostic, retain_graph=True)
    result = dict(
        arm=arm,
        status="complete",
        graph_hash=canonical_hash(graph),
        preparation_hash=preparation.content_hash,
        graph_nodes=len(graph.nodes),
        graph_edges=len(graph.edges),
        omitted_nodes=len(graph.omitted_nodes),
        omitted_supports=len(graph.omitted_supports),
        omitted_evidence=list(graph.omitted_evidence),
        support_omissions=graph.support_omissions,
        selected_pairs=len(pairs.pairs),
        pair_factors=len(factors),
        pair_omissions=pairs.omissions,
        graph_schema_hash=model.graph_schema_hash,
        parameter_count=sum(p.numel() for p in model.parameters()),
        readout_gradients=readouts,
        proposals=[],
        optimizer_steps=0,
        checkpoints_written=0,
    )
    write_artifact(output / "progress.json", result)
    proposal_deadline = time.monotonic() + settings["proposal_seconds"]
    for obj in problem.objects:
        menu = retrieval.for_object(obj.object_id)
        remaining = proposal_deadline - time.monotonic()
        if remaining <= 0:
            result["proposals"].append(dict(object_id=obj.object_id, status="proposal_deadline"))
            continue
        grammar = with_immutable_context(
            mapping_grammar(
                obj,
                menu.classes,
                menu.properties,
                **settings["grammar"],
                fixed_axioms=problem.fixed_axioms,
                source_classes=menu.source_classes,
                target_classes=menu.target_classes,
                source_properties=menu.source_properties,
                target_properties=menu.target_properties,
                constraint_identity=canonical_hash((problem.policy, menu)),
            ),
            problem.fixed_axioms,
            problem.policy,
        )
        # Unexpected compiler/software exceptions propagate to a durable failed row.
        distribution = proposal_distribution(
            model,
            memory,
            obj,
            encoding=grammar,
            compile_seconds=min(settings["compile_seconds"], remaining),
            **settings["proposal"],
        )
        if not getattr(distribution.circuit, "complete", True):
            result["proposals"].append(dict(object_id=obj.object_id, status="partial_circuit"))
            continue
        probabilities = torch.stack(
            [distribution.candidate_log_probability(c) for c in obj.candidates]
        )
        finite = torch.isfinite(probabilities)
        if not finite.any():
            result["proposals"].append(
                dict(object_id=obj.object_id, status="unavailable_inventory")
            )
            continue
        # Uniform inventory log likelihood is a numerical diagnostic, not a semantic target.
        gradients = gradient_summary(model, -probabilities[finite].mean(), retain_graph=True)
        result["proposals"].append(
            dict(
                object_id=obj.object_id,
                status="complete",
                candidates=len(obj.candidates),
                covered_candidates=int(finite.sum()),
                missing_candidates=int((~finite).sum()),
                variable_count=grammar.variable_count,
                gradients=gradients,
            )
        )
        write_artifact(output / "progress.json", result)
    if device == "cuda":
        torch.cuda.synchronize()
        result.update(
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            device=torch.cuda.get_device_name(0),
        )
    result["elapsed_seconds"] = time.monotonic() - started
    if (
        result["omitted_nodes"]
        or result["omitted_supports"]
        or result["omitted_evidence"]
        or result["support_omissions"]
        or result["pair_omissions"]
        or any(
            p["status"] != "complete" or p.get("missing_candidates", 0) for p in result["proposals"]
        )
    ):
        result["status"] = "partial"
    write_artifact(output / "progress.json", result)
    return result


def run(plan_path, arm, output):
    from exact.repair.workers import bounded_call
    from tools.repair.numerical_profile import probe_identity

    plan = read(plan_path)
    rows = schedule(plan)
    if arm not in ARMS:
        raise ValueError("Unknown probe arm")
    declaration = verify_binding(plan["declaration"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash((probe_identity(plan_path, plan_path), sha(__file__), arm))
    with (output / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = []
        for item in rows:
            key = canonical_hash(item)
            row_identity = canonical_hash((identity, key))
            target, pending = output / "rows" / (key + ".json"), output / "pending" / (
                key + ".json"
            )
            row = checked_checkpoint(target, row_identity)
            if row is None:
                if pending.exists():
                    raise RuntimeError(
                        "Unfinished probe requires prior-owner/cleanup reconciliation; no silent rerun"
                    )
                record = verify_binding(item["observable"])
                if read_record(record).content_hash != item["input_hash"]:
                    raise ValueError("Development observable identity changed")
                checkpoint(pending, row_identity, started_epoch=time.time(), case=item)
                result = bounded_call(
                    probe,
                    record,
                    plan["settings"],
                    declaration,
                    arm,
                    str(output / "payloads" / key),
                    **plan["row_budget"],
                )
                row = checkpoint(
                    target,
                    row_identity,
                    case=item,
                    call_status=result.status,
                    detail=result.detail,
                    result=result.value,
                    cleanup_complete=result.cleanup_complete,
                    resources=dict(result.resource_usage),
                )
            receipts.append(binding(target))
            write_artifact(
                output / "progress.json",
                dict(arm=arm, recorded_rows=len(receipts), scheduled_rows=32),
            )
            if not row["cleanup_complete"] or row["call_status"] == "error":
                raise RuntimeError(
                    "Probe failure retained in " + str(target) + ": " + str(row.get("detail", ""))
                )
        outcomes = [read(ref["path"]) for ref in receipts]
        call_statuses = dict(Counter(row["call_status"] for row in outcomes))
        probe_statuses = dict(
            Counter((row.get("result") or {}).get("status", "unavailable") for row in outcomes)
        )
        report = checkpoint(
            output / "report.json",
            identity,
            schema="exact-repair/training-probe/v1",
            status="complete",
            arm=arm,
            rows=receipts,
            scheduled_rows=32,
            recorded_rows=len(receipts),
            plan=binding(plan_path),
            call_statuses=call_statuses,
            probe_statuses=probe_statuses,
            qualified_rows=sum(
                row["call_status"] == "complete"
                and (row.get("result") or {}).get("status") == "complete"
                for row in outcomes
            ),
            optimizer_steps=0,
            model_fitting=False,
            evaluator_records_opened=False,
            heldout_cases_opened=False,
            gates_established=False,
            limitations=[
                "Observable numerical gradients only; no admitted semantic supervision or convergence.",
                "Per-case fresh weights and single-case memory; no minibatch or Adam-state capacity claim.",
                "Unknown/partial rows retain their scheduled denominator.",
                "Generated acquisition, decoded development selection and18fitting protocols remain required.",
            ],
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.arm, args.output)


if __name__ == "__main__":
    main()
