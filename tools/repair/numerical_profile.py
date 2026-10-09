"""Development-only, failure-inclusive graph/teacher/CUDA resource measurements.

The numerical loss is a diagnostic, not semantic training. Every case/arm starts
from fresh weights. No pilot checkpoint or held-out outcome is read.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import time
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound, immutable, validate_profile
from tools.repair.expanded_profile import checked_checkpoint, checkpoint, parent_fingerprints
from tools.repair.prepare import case_from_dict

ARMS = tuple(
    f"{encoder}-{head}" for encoder in ("hgt", "rgcn", "none") for head in ("unary", "pairwise")
)
STAGES = ("graph", "teacher", *ARMS)


def validate_gpu(plan):
    complete = bound(plan["gpu_completion"])
    batch = bound(plan["gpu_batch"])
    outputs = bound(plan["gpu_outputs"])
    report = bound(plan["gpu_report"])
    if (
        complete["status"] != "complete"
        or complete["exit_code"] != 0
        or complete["batch"] != plan["gpu_batch"]["path"]
        or (complete["step_id"], complete["dispatch_nonce"])
        != (plan["gpu_step"], plan["gpu_nonce"])
        or report["status"] != "complete"
        or {r["encoder"] for r in report["rows"]} != {"hgt", "rgcn", "none"}
        or outputs["gpu-qualification.json"] != plan["gpu_report"]["sha256"]
        or Path(complete["work"]) / "gpu-qualification.json" != Path(plan["gpu_report"]["path"])
    ):
        raise ValueError("GPU qualification ownership or coverage mismatch")
    for path, digest in batch["frozen_files"].items():
        if sha(path) != digest:
            raise ValueError("GPU qualification source changed: " + path)


def schedule(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    if plan["schema"] != "exact-repair/numerical-profile-plan/v1":
        raise ValueError("Unknown numerical profile plan")
    validate_gpu(plan)
    _, inventory, summary = validate_profile(plan)
    base = Path(plan["profile_report"]["path"]).parent
    rows = []
    for parent in inventory["selected"]:
        if parent["split"] != "profile":
            continue
        for control in ("corrupted", "coherent"):
            key = canonical_hash((parent["key"], control))
            source = binding(base / "development_cases" / (key + ".json"))
            case = case_from_dict(bound(source))
            if (
                case.split != "development"
                or case.schema_revision != "v3"
                or case.structural_parent != parent["key"]
                or case.control != control
                or not set(parent_fingerprints(case)) <= set(parent["fingerprints"])
            ):
                raise ValueError("Numerical profile case crosses frozen development scope")
            rows.append(
                dict(
                    key=key,
                    parent=parent["key"],
                    family=parent["family"],
                    control=control,
                    case=source,
                    case_hash=bound(source)["hash"],
                )
            )
    for ordinal, parent in enumerate(inventory["missing"]):
        if parent["split"] == "profile":
            for control in ("corrupted", "coherent"):
                rows.append(
                    dict(
                        key=canonical_hash((parent, ordinal, control)),
                        parent=None,
                        family=parent["family"],
                        control=control,
                        case=None,
                        status="unavailable_structural_parent",
                    )
                )
    if len(rows) != 32 or len({r["key"] for r in rows}) != 32:
        raise ValueError("Numerical profile requires the full 32-case denominator")
    return dict(
        schema="exact-repair/numerical-profile-schedule/v1",
        plan=binding(plan_path),
        cases=rows,
        stages=STAGES,
        cases_per_stage=32,
        numerical_rows=192,
        native_profile=summary,
        test_outcomes_opened=False,
        purpose="Development resource evidence; no semantic training or gate passage",
    )


def load_schedule(plan_path, schedule_path):
    frozen = read(schedule_path)
    if canonical_hash(frozen) != canonical_hash(schedule(plan_path)):
        raise ValueError("Frozen numerical profile schedule changed")
    return read(plan_path), frozen


def prepare_graph(record, settings):
    from exact.repair.graph import EffectivePreparation, feature_vector
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    case = case_from_dict(record)
    if case.split != "development" or case.schema_revision != "v3":
        raise ValueError("Only native v3 development cases may be profiled")
    started = time.monotonic()
    preparation = EffectivePreparation(
        **settings["graph"], retrieval_config=RetrievalConfig(**settings["retrieval"])
    )
    retrieved = retrieve_vocabulary(case.problem, config=preparation.retrieval_config)
    retrieval_seconds = time.monotonic() - started
    started = time.monotonic()
    graph = preparation.graph(case.problem, retrieved)
    graph_seconds = time.monotonic() - started
    started = time.monotonic()
    features = [feature_vector(n, settings["model"]["feature_dim"]) for n in graph.nodes]
    feature_seconds = time.monotonic() - started
    pairs = preparation.pairs(case.problem, graph)
    partial = bool(
        graph.omitted_nodes
        or graph.omitted_supports
        or graph.omitted_evidence
        or graph.support_omissions
    )
    result = dict(
        status="partial" if partial else "complete",
        graph_hash=canonical_hash(graph),
        graph_nodes=len(graph.nodes),
        graph_edges=len(graph.edges),
        metadata=graph.metadata,
        retrieval_seconds=retrieval_seconds,
        graph_seconds=graph_seconds,
        feature_seconds=feature_seconds,
        feature_values=sum(map(len, features)),
        omitted_nodes=len(graph.omitted_nodes),
        omitted_supports=len(graph.omitted_supports),
        omitted_evidence=list(graph.omitted_evidence),
        support_omissions=graph.support_omissions,
        selected_pairs=len(pairs.pairs),
        candidate_factors=pairs.candidate_factors,
        pair_omissions=pairs.omissions,
        candidate_counts=[len(obj.candidates) for obj in case.problem.objects],
        preparation_identity=preparation.content_hash,
    )
    return case, graph, pairs, result


def graph_probe(record, settings, directory, context):
    return prepare_graph(record, settings)[3]


def teacher_probe(record, settings, directory, context):
    from tools.repair.prepare import publish_label_cache
    from tools.repair.train import label_case

    case = case_from_dict(record)
    if case.split != "development":
        raise ValueError("Teacher probe cannot open held-out cases")
    cache = label_case(case, **settings["teacher"])
    artifact = publish_label_cache(cache, Path(directory))
    return dict(
        status="complete" if cache.complete else "partial",
        coverage=cache.coverage,
        stop_reason=cache.stop_reason,
        teacher_seconds=cache.elapsed_seconds,
        artifacts=[artifact],
        scope="bounded fixed-inventory teacher timing, not generated-pool labels or training data",
    )


def diagnostic_loss(model, case, graph, pairs):
    """Exercise actual differentiable readouts without inventing semantic labels."""
    import torch

    memory = model.encode(graph)
    unary, factors = model.score_inventory(case.problem.objects, memory, interaction_pairs=pairs)
    risk = model.plan_risk_logit(
        case.problem.objects,
        memory,
        tuple(0 for _ in case.problem.objects),
        supports=graph.admitted_supports,
    )
    terms = (
        [value.square().mean() for value in unary]
        + [value.square() for value in factors.values()]
        + [risk.square()]
    )
    return torch.stack(terms).mean(), len(factors)


def numerical_probe(record, settings, directory, context):
    import torch
    from exact.repair.model import RepairModel
    from tools.repair.train import save_training_state

    case, graph, pairs, graph_result = prepare_graph(record, settings)
    if graph_result["graph_hash"] != context["graph_hash"]:
        raise ValueError("GPU graph differs from the frozen CPU preparation")
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one allocated CUDA device is required")
    if torch.cuda.get_device_name(0) != settings["device_name"]:
        raise ValueError("Unexpected GPU for the frozen profile")
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    encoder, head = context["arm"].split("-")
    pairwise = head == "pairwise"
    started = time.monotonic()
    metadata = (context["metadata"][0], tuple(map(tuple, context["metadata"][1])))
    model = RepairModel(metadata, **settings["model"], encoder=encoder, pairwise=pairwise).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), **settings["optimizer"])
    torch.cuda.synchronize()
    initialization_seconds = time.monotonic() - started
    samples = []
    for iteration in range(settings["iterations"]):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        loss, factor_count = diagnostic_loss(model, case, graph, pairs.pairs if pairwise else ())
        torch.cuda.synchronize()
        forward_seconds = time.monotonic() - started
        started = time.monotonic()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.monotonic() - started
        gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        if (
            not gradients
            or not torch.isfinite(loss).item()
            or not all(torch.isfinite(g).all().item() for _, g in gradients)
        ):
            raise ValueError("Nonfinite or missing diagnostic gradients")
        started = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        sample = dict(
            iteration=iteration,
            phase="warmup" if iteration == 0 else "measured",
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            optimizer_seconds=time.monotonic() - started,
            diagnostic_loss=loss.item(),
            pair_factors=factor_count,
            parameters_with_gradients=len(gradients),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        samples.append(sample)
        write_artifact(Path(directory) / "progress.json", dict(samples=samples, graph=graph_result))
    saved = Path(directory) / "profiling-only.pt"
    save_training_state(
        saved,
        dict(
            schema="exact-repair/profiling-only-weights/v1",
            use_for_training=False,
            arm=context["arm"],
            case_hash=record["hash"],
            settings_hash=canonical_hash(settings),
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            completed_iterations=len(samples),
        ),
    )
    return dict(
        status=graph_result["status"],
        samples=samples,
        graph=graph_result,
        initialization_seconds=initialization_seconds,
        parameter_count=sum(p.numel() for p in model.parameters()),
        artifacts=[binding(saved)],
        device=torch.cuda.get_device_name(0),
        pair_support=(
            "measured" if pairwise and pairs.pairs else "no_selected_pairs" if pairwise else "unary"
        ),
        scope="Diagnostic squared readout loss with fresh disposable weights; no supervised optimization, proposal circuit, active collection or decoder",
    )


def probe_identity(plan_path, schedule_path):
    from exact.repair.study import runtime_manifest

    runtime = runtime_manifest()
    return canonical_hash(
        (
            sha(plan_path),
            sha(schedule_path),
            sha(__file__),
            runtime["dependencies"],
            runtime["ontology_implementations"],
            runtime["code_hashes"],
        )
    )


def validate_artifacts(row):
    for artifact in (row.get("result") or {}).get("artifacts", []):
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Probe artifact changed: " + artifact["path"])


def read_stage(output, identity, stage, scheduled):
    report = checked_checkpoint(Path(output) / "report.json", canonical_hash((identity, stage)))
    if (
        report is None
        or report["stage"] != stage
        or report["scheduled"] != len(scheduled)
        or report["status"] != "complete"
        or report["test_outcomes_opened"]
    ):
        raise ValueError("Missing or incompatible stage completion: " + stage)
    expected = {item["key"] for item in scheduled}
    rows = []
    for item in report["rows"]:
        row = bound(item)
        if row["key"] not in expected:
            raise ValueError("Duplicate or unexpected stage case")
        expected.remove(row["key"])
        checked_checkpoint(Path(item["path"]), canonical_hash((identity, stage, row["key"])))
        if not row["cleanup_complete"]:
            raise ValueError("Unconfirmed native worker cleanup")
        validate_artifacts(row)
        rows.append(row)
    if expected:
        raise ValueError("Stage lost scheduled cases")
    return report, rows


def run_stage(plan_path, schedule_path, stage, output, graph_output=None):
    from exact.repair.workers import bounded_call

    if stage not in STAGES:
        raise ValueError("Unknown probe stage")
    plan, frozen = load_schedule(plan_path, schedule_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = probe_identity(plan_path, schedule_path)
        context, graphs = {}, {}
        if stage in ARMS:
            _, graph_rows = read_stage(graph_output, identity, "graph", frozen["cases"])
            graphs = {r["key"]: r for r in graph_rows}
            metadata = [set(), set()]
            for row in graph_rows:
                result = row.get("result")
                if row["call_status"] == "complete" and result:
                    metadata[0].update(result["metadata"][0])
                    metadata[1].update(map(tuple, result["metadata"][1]))
            context = dict(arm=stage, metadata=[sorted(metadata[0]), sorted(metadata[1])])
        kind = "numerical" if stage in ARMS else stage
        function = {"graph": graph_probe, "teacher": teacher_probe, "numerical": numerical_probe}[
            kind
        ]
        budget = plan["budgets"][kind]
        receipts = []
        for case in frozen["cases"]:
            path = output / "rows" / (case["key"] + ".json")
            row_identity = canonical_hash((identity, stage, case["key"]))
            row = checked_checkpoint(path, row_identity)
            if row is None:
                pending = output / "pending" / (case["key"] + ".json")
                payload = output / "payloads" / case["key"]
                common = dict(
                    key=case["key"],
                    case=case,
                    cleanup_complete=True,
                    elapsed_seconds=0,
                    resources={},
                )
                if pending.exists():
                    # A lock proves only coordinator ownership, not native child cleanup.
                    # Never silently repeat a fixed-budget probe or invent cleanup proof.
                    checked_checkpoint(pending, row_identity)
                    raise RuntimeError(
                        "Unfinished fixed-budget probe: reconcile prior owner, cleanup and partial payload before continuation"
                    )
                elif case["case"] is None:
                    row = checkpoint(
                        path,
                        row_identity,
                        **common,
                        call_status="unavailable_structural_parent",
                        result=None,
                    )
                elif stage in ARMS and (
                    graphs[case["key"]]["call_status"] != "complete"
                    or not graphs[case["key"]].get("result")
                ):
                    row = checkpoint(
                        path, row_identity, **common, call_status="unavailable_graph", result=None
                    )
                else:
                    record = bound(case["case"])
                    if (
                        record["hash"] != case["case_hash"]
                        or case_from_dict(record).split != "development"
                    ):
                        raise ValueError("Frozen development case identity changed")
                    if stage in ARMS:
                        context["graph_hash"] = graphs[case["key"]]["result"]["graph_hash"]
                    checkpoint(pending, row_identity, budget=budget, started_epoch=time.time())
                    write_artifact(
                        output / "progress.json",
                        dict(stage=stage, recorded=len(receipts), case=case["key"]),
                    )
                    started = time.monotonic()
                    result = bounded_call(
                        function,
                        record,
                        plan["settings"],
                        str(payload),
                        context,
                        timeout=budget["seconds"],
                        memory_mb=budget["memory_mb"],
                    )
                    common.update(
                        elapsed_seconds=time.monotonic() - started,
                        resources=dict(result.resource_usage),
                        cleanup_complete=result.cleanup_complete,
                    )
                    row = checkpoint(
                        path,
                        row_identity,
                        **common,
                        call_status=result.status,
                        detail=result.detail,
                        result=result.value,
                    )
            if not row["cleanup_complete"]:
                raise RuntimeError(
                    "Native worker cleanup incomplete; inspect ownership before continuing"
                )
            validate_artifacts(row)
            receipts.append(binding(path))
        report = checkpoint(
            output / "report.json",
            canonical_hash((identity, stage)),
            schema="exact-repair/numerical-profile-stage/v1",
            stage=stage,
            scheduled=len(frozen["cases"]),
            rows=receipts,
            status="complete",
            test_outcomes_opened=False,
            training_complete=False,
        )
        write_artifact(output / "progress.json", dict(stage="complete", recorded=len(receipts)))
        return report


def summarize(plan_path, schedule_path, outputs, output):
    plan, frozen = load_schedule(plan_path, schedule_path)
    if set(outputs) != set(STAGES):
        raise ValueError("Every registered profiling stage must be accounted for")
    identity = probe_identity(plan_path, schedule_path)
    stages, numerical_times, peaks = {}, [], []
    for stage in STAGES:
        _, rows = read_stage(outputs[stage], identity, stage, frozen["cases"])
        counts = Counter(
            r["call_status"] + "/" + (r.get("result") or {}).get("status", "unknown") for r in rows
        )
        stages[stage] = dict(
            report=binding(Path(outputs[stage]) / "report.json"),
            scheduled=32,
            outcomes=dict(counts),
            elapsed_seconds=sum(r["elapsed_seconds"] for r in rows),
        )
        if stage in ARMS:
            for row in rows:
                for sample in (row.get("result") or {}).get("samples", []):
                    peaks.append(sample["peak_cuda_reserved_bytes"])
                    if sample["phase"] == "measured":
                        numerical_times.append(
                            sum(
                                sample[k]
                                for k in (
                                    "forward_seconds",
                                    "backward_seconds",
                                    "optimizer_seconds",
                                )
                            )
                        )
    numerical_times.sort()
    p95 = numerical_times[math.ceil(0.95 * len(numerical_times)) - 1] if numerical_times else None
    recommendations = dict(
        operational_slice_seconds=7200,
        checkpoint="after each case and acquisition/development phase; persist optimizer/RNG",
        batch_cases=1,
        worker=dict(cpus=4, gpus=1, memory_mb=32768),
        numerical_step_p95_seconds=p95,
        peak_cuda_reserved_bytes=max(peaks, default=None),
        numerical_only_128_case_epoch_estimate_seconds=None if p95 is None else 128 * p95,
        basis="Development stage timings only; acquisition, circuit gradients, decoder and validation are excluded. Two-hour resumable operational slices are provisional, not a campaign ceiling or scientific per-case budget.",
        scientific_budget_change=False,
        full_training_runtime_estimate=None,
        missing_components=[
            "generated-pool teacher acquisition",
            "proposal circuit backward",
            "generated development selection",
            "50-epoch convergence",
        ],
    )
    return checkpoint(
        Path(output),
        canonical_hash((identity, "summary")),
        schema="exact-repair/numerical-profile-completion/v1",
        status="complete",
        stages=stages,
        numerical_denominator=192,
        graph_denominator=32,
        teacher_denominator=32,
        native_profile=frozen["native_profile"],
        recommendations=recommendations,
        test_outcomes_opened=False,
        training_complete=False,
        gates_passed=False,
        interpretation="All resource probes accounted for, including missing/partial/unknown. Fixed-inventory numerical diagnostics do not establish generated-pool quality or end-to-end training capacity.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-schedule", "run", "summarize"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--graph-output", type=Path)
    parser.add_argument("--outputs", type=Path)
    args = parser.parse_args()
    if args.command == "freeze-schedule":
        immutable(args.schedule, schedule(args.plan))
    elif args.command == "run":
        run_stage(args.plan, args.schedule, args.stage, args.output, args.graph_output)
    else:
        summarize(args.plan, args.schedule, read(args.outputs), args.output)


if __name__ == "__main__":
    main()
