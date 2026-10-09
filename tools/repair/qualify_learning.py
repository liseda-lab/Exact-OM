"""Full-schema optimizer and exact checkpoint qualification on one allocated device.

Typed fixtures exercise every declared relation; the generated repair fixture
also exercises value/risk readouts. This is engineering evidence, not a fit.
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash


def run(
    output,
    *,
    device="cuda",
    encoder="hgt",
    width=128,
    layers=2,
    heads=4,
    seed=13,
    prepared=None,
    seconds=600,
):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from exact.repair.graph import (
        FEATURE_SCHEMA_V3,
        GraphNode,
        ObservableGraph,
        EffectivePreparation,
    )
    from exact.repair.graph_schema import declared_metadata, generic_graph_schema
    from exact.repair.model import RepairModel, repair_benefits
    from exact.repair.retrieval import retrieve_vocabulary
    from tools.repair.corpus import generate_corpus
    from tools.repair.train import save_training_state

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA qualification has no visible allocated GPU")
    started = time.monotonic()
    schema = generic_graph_schema()
    metadata = declared_metadata(schema)
    case = generate_corpus(
        split_counts={"train": 1, "development": 0, "test": 0},
        siblings_per_parent=1,
        families=("interaction_complementary",),
        revision="v3",
    )[0]
    preparation = EffectivePreparation(revision="v3")
    base = preparation.graph(case.problem, retrieve_vocabulary(case.problem))
    typed = tuple(GraphNode("schema:" + kind, kind, (("observed", 1.0),)) for kind in metadata[0])
    edges = tuple(("schema:" + a, r, "schema:" + b) for a, r, b in metadata[1])
    graph = ObservableGraph(
        base.nodes + typed, base.edges + edges, base.object_nodes, feature_schema=FEATURE_SCHEMA_V3
    )
    model = RepairModel(
        metadata,
        graph_schema=schema,
        encoder=encoder,
        hidden_dim=width,
        heads=heads,
        layers=layers,
        dropout=0.1,
        revision="v3",
        pairwise=True,
    ).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    pairs = preparation.pairs(case.problem, graph).pairs
    assignments = [
        tuple(0 for _ in case.problem.objects),
        tuple(len(obj.candidates) - 1 for obj in case.problem.objects),
    ]

    def update(m, optim):
        optim.zero_grad(set_to_none=True)
        memory = m.encode(graph)
        unary, factors = m.score_inventory(case.problem.objects, memory, interaction_pairs=pairs)
        values = repair_benefits(assignments, unary, factors)
        risk = torch.stack(
            [
                m.plan_risk_logit(case.problem.objects, memory, assignment)
                for assignment in assignments
            ]
        )
        loss = (
            (values - values.new_tensor([0.0, 0.7])).square().mean()
            + torch.nn.functional.binary_cross_entropy_with_logits(
                risk, risk.new_tensor([0.0, 1.0])
            )
            + sum(row.square().mean() for row in memory.rows.values()) / len(memory.rows)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite actual optimizer qualification loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        if device == "cuda":
            torch.cuda.synchronize()
        return float(loss.detach())

    first = update(model, optimizer)
    save_training_state(
        output / "qualification-state.pt",
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            updates=1,
            graph_schema_hash=canonical_hash(schema),
            config=model.config,
            metadata=model.metadata,
        ),
    )
    second = update(model, optimizer)
    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    saved = torch.load(output / "qualification-state.pt", weights_only=True, map_location=device)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"].cpu())
    if device == "cuda":
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    resumed_loss = update(model, optimizer)
    exact = all(
        torch.equal(value.detach().cpu(), expected[key])
        for key, value in model.state_dict().items()
    )
    model.eval()
    with torch.no_grad():
        memory = model.encode(base)
        unary, factors = model.score_inventory(
            case.problem.objects, memory, interaction_pairs=pairs
        )
        prediction = repair_benefits(assignments, unary, factors)
    actual_rows = []
    max_allocated = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    max_reserved = torch.cuda.max_memory_reserved() if device == "cuda" else 0
    if prepared is not None:
        from tools.repair.prepare import load_preparation

        cases, _, _ = load_preparation(Path(prepared))
        if any(case.split != "development" for case in cases):
            raise ValueError(
                "Hardware qualification preparation must contain exposed DEV cases only"
            )
        model.train()
        for actual_case in cases:
            row = dict(
                case_id=actual_case.case_id,
                family=actual_case.family,
                control=actual_case.control,
                input_hash=actual_case.problem.content_hash,
                status="not_attempted_deadline",
                optimizer_updates=0,
            )
            if time.monotonic() - started < seconds:
                try:
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    graph = preparation.graph(
                        actual_case.problem, retrieve_vocabulary(actual_case.problem)
                    )
                    case = actual_case
                    pairs = preparation.pairs(case.problem, graph).pairs
                    assignments = [
                        tuple(0 for _ in case.problem.objects),
                        tuple(len(obj.candidates) - 1 for obj in case.problem.objects),
                    ]
                    row.update(
                        loss=update(model, optimizer),
                        status="complete",
                        optimizer_updates=1,
                        graph_nodes=len(graph.nodes),
                        graph_edges=len(graph.edges),
                        device=device,
                        nonfinite=False,
                        candidate_counts=[len(obj.candidates) for obj in case.problem.objects],
                    )
                except (RuntimeError, ValueError, KeyError) as error:
                    row.update(status="unavailable", detail=f"{type(error).__name__}: {error}")
                finally:
                    row["max_cuda_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated() if device == "cuda" else 0
                    )
                    row["max_cuda_reserved_bytes"] = (
                        torch.cuda.max_memory_reserved() if device == "cuda" else 0
                    )
                    max_allocated = max(max_allocated, row["max_cuda_allocated_bytes"])
                    max_reserved = max(max_reserved, row["max_cuda_reserved_bytes"])
            actual_rows.append(row)
            write_artifact(
                output / "actual-case-progress.json",
                dict(
                    expected_cases=len(cases),
                    rows=actual_rows,
                    remaining=len(cases) - len(actual_rows),
                ),
            )
    report = dict(
        schema="exact-repair/full-schema-optimizer-qualification/v1",
        status="qualified" if exact and bool(torch.isfinite(prediction).all()) else "failed",
        graph_schema_hash=canonical_hash(schema),
        node_types=len(metadata[0]),
        relations=len(metadata[1]),
        encoder=encoder,
        width=width,
        layers=layers,
        heads=heads,
        device=device,
        gpu_name=torch.cuda.get_device_name() if device == "cuda" else None,
        parameter_count=sum(p.numel() for p in model.parameters()),
        first_loss=first,
        second_loss=second,
        resumed_loss=resumed_loss,
        exact_resume=exact,
        optimizer_updates=2,
        actual_optimizer_states=len(optimizer.state),
        changed_parameters=sum(
            not torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items()
        ),
        max_cuda_allocated_bytes=max_allocated,
        max_cuda_reserved_bytes=max_reserved,
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_case_rows=actual_rows,
        actual_cases_scheduled=len(actual_rows),
        actual_cases_completed=sum(r["status"] == "complete" for r in actual_rows),
        prepared=str(prepared) if prepared else None,
        seconds=seconds,
        elapsed_seconds=time.monotonic() - started,
        fitting_data_used=False,
        held_out_opened=False,
    )
    if any(row["status"] != "complete" for row in actual_rows):
        report["status"] = "incomplete_actual_case_qualification"
    write_artifact(output / "report.json", report)
    if report["status"] != "qualified":
        raise RuntimeError("Actual full-schema optimizer/resume qualification failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--encoder", choices=("hgt", "rgcn", "none"), default="hgt")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args()
    run(
        args.output,
        device=args.device,
        encoder=args.encoder,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        seed=args.seed,
        prepared=args.prepared,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    main()
