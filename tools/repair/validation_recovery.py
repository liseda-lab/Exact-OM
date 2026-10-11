"""Re-evaluate authenticated saved weights; no optimizer or TEST entry point.

Original training states and DEV rows are never changed. The manifest binds each
recovery's code, weights, cases and budget. Completed rows are reusable only under
that same identity; begun cases retain their original deadlines after a restart.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable


def weight_digest(values):
    import hashlib
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_hash((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot(source, primary_manifest, risk_directory, output):
    """Match immutable risk weights to an authenticated terminal training state."""
    import torch
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest

    torch.set_num_threads(1)
    state = torch.load(authenticate(source), map_location="cpu", weights_only=True)
    manifest = bound(primary_manifest)
    protocol = bound(manifest["protocol"])
    if state.get("optimizer_updates") != 756 or len(state.get("history", [])) != 6:
        raise ValueError("Expected the completed common six-epoch endpoint")
    expected = weight_digest(state["model"])
    matches = []
    for path in sorted(Path(risk_directory).glob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if weight_digest(saved["state_dict"]) != expected:
            continue
        cfg, arch = saved["config"], protocol["model"]
        required = dict(
            encoder=arch["backbone"],
            hidden_dim=arch["hidden_width"],
            layers=arch["layers"],
            heads=arch["attention_heads"],
            graph_schema=arch["graph_schema"],
            pairwise=arch["pair_benefit"],
            support_enabled=arch["support_enabled"],
            revision="v3",
        )
        if (
            any(cfg.get(k) != v for k, v in required.items())
            or saved["metadata"] != state["metadata"]
        ):
            raise ValueError("Saved snapshot architecture/metadata differs from trained model")
        model = RepairModel(saved["metadata"], **cfg)
        model.load_state_dict(saved["state_dict"], strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("Nonfinite saved weights")
        if sha(path) != path.stem:
            raise ValueError("Content-addressed risk snapshot was changed")
        matches.append(dict(**binding(path), model_hash=model_digest(model)))
    if len(matches) != 1:
        raise ValueError("Expected exactly one byte-identical saved model snapshot")
    value = dict(
        schema="exact-repair/validation-weight-recovery/v1",
        source=source,
        primary_manifest=primary_manifest,
        artifact=matches[0],
        epoch=6,
        seed=manifest["seed"],
        condition=manifest["condition"],
        optimizer_updates=756,
        weights_unchanged=True,
        selected=False,
    )
    immutable(Path(output), value)
    return value


def qualification(rows, expected_families):
    """Operational gate, never a threshold on learned versus symbolic quality."""
    verified = [
        r
        for r in rows
        if r["result"].get("decoded", {}).get("logical_status") == "VERIFIED_FEASIBLE"
    ]
    repaired = {r["family"] for r in verified if r["control"] == "corrupted"}
    cleanup = [
        r["case_id"]
        for r in rows
        if "CLEANUP" in r["result"].get("status", "").upper()
        or r["result"].get("cleanup_complete") is False
    ]
    missing = sorted(set(expected_families) - repaired)
    return dict(
        passed=not cleanup and not missing,
        missing_repair_families=missing,
        cleanup_failures=cleanup,
        verified_controls=dict(Counter(r["control"] for r in verified)),
        statuses=dict(Counter(r["result"]["status"] for r in rows)),
        performance_comparison_required=False,
    )


def runtime_configuration(protocol):
    """Use exactly the primary entry point's defaults for omitted optional keys."""
    import inspect
    from tools.repair import train

    config = {
        name: value.default
        for name, value in inspect.signature(train.train_cases).parameters.items()
        if value.default is not inspect.Parameter.empty
    }
    config.update(train._protocol_arguments(protocol))
    config["profile"] = train._protocol_profile(protocol)
    return config


def run(manifest_path, output, *, gpu_uuid=None):
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.learning import SemanticTargetSpec
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest
    from exact.repair.protocol import load_protocol_v3, training_projection_v3
    from exact.repair.retrieval import retrieve_vocabulary
    from tools.repair import train
    from tools.repair.prepare import load_preparation
    from tools.repair.primary_runtime import owned_device, scheduled_runtime_development
    from exact.repair.api import write_artifact

    manifest, output = read(manifest_path), Path(output)
    if (
        manifest.get("schema") != "exact-repair/validation-recovery/v1"
        or manifest.get("test_allowed") is not False
        or manifest.get("optimizer_updates") != 0
    ):
        raise ValueError("Recovery admits only saved-weight DEV")
    owner = gpu_uuid or manifest["gpu_uuid"]
    actual = owned_device(owner)
    identity = binding(Path(manifest_path))
    immutable(output / "manifest-binding.json", identity)
    recovery = bound(manifest["weights"])
    primary = bound(recovery["primary_manifest"])
    protocol = training_projection_v3(
        load_protocol_v3(authenticate(primary["protocol"]), for_execution=True)
    )
    cases, caches, _ = load_preparation(authenticate(primary["preparation"]))
    config = runtime_configuration(protocol)
    dev = {
        case.case_id: (case, cache)
        for case, cache in scheduled_runtime_development(cases, caches, config)
    }
    ids = manifest["case_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frozen DEV case")
    if manifest.get("case_gpu_uuids") is not None:
        if (
            set(manifest["case_gpu_uuids"]) != set(ids)
            or owner not in manifest["case_gpu_uuids"].values()
        ):
            raise ValueError("Incomplete or unowned hardware assignment")
        ids = [key for key in ids if manifest["case_gpu_uuids"][key] == owner]
    elif owner != manifest["gpu_uuid"]:
        raise ValueError("Undeclared GPU override")
    if len(ids) != len(set(ids)) or any(
        key not in dev or dev[key][0].split != "development" for key in ids
    ):
        raise ValueError("Case identity or DEV boundary changed")
    torch.set_num_threads(1)
    saved = torch.load(authenticate(recovery["artifact"]), weights_only=True, map_location="cpu")
    model = RepairModel(saved["metadata"], **saved["config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    if model_digest(model) != recovery["artifact"]["model_hash"]:
        raise ValueError("Recovered model identity changed")
    model.to("cuda").eval()
    del saved
    preparation = EffectivePreparation(
        **{
            k: config[k]
            for k in (
                "max_graph_nodes",
                "max_graph_edges",
                "max_explanations",
                "max_text_tokens",
                "pair_factor_limit_per_object",
                "pair_max_pairs",
                "pair_max_factors",
                "retrieval_config",
                "revision",
            )
        }
    )
    options = {
        k: config[k]
        for k in (
            "selection_options",
            "case_cpu_seconds",
            "execution_schedule",
            "circuit_limits",
            "compiler_cache_directory",
            "vtree_type",
            "candidate_cap",
            "mixtures",
            "profile",
            "pair_factor_limit_per_object",
            "pair_max_pairs",
            "pair_max_factors",
            "max_depth",
            "max_constructors",
            "max_circuit_nodes",
            "compile_seconds",
            "max_graph_nodes",
            "max_graph_edges",
            "max_explanations",
            "max_text_tokens",
            "quantization_scale",
            "retrieval_config",
            "target_basis",
            "mixture_symbolic_weight",
            "development_use_policy",
        )
    }
    options.update(
        plan_rating_aggregation=config.get("plan_rating_aggregation"),
        draws_per_object=config["development_draws_per_object"],
        seed=recovery["seed"],
        temperature=config["proposal_temperature"],
        proposal_arm=config["proposal_arm"],
        elementary_seconds=manifest["elementary_seconds"],
        device="cuda",
        model_snapshot=recovery["artifact"],
    )
    deadline = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"])
    rows = []
    for case_id in ids:
        case, cache = dev[case_id]
        slot = dict(
            seed=recovery["seed"],
            supervision_condition=recovery["condition"],
            epoch=recovery["epoch"],
            case_id=case_id,
        )
        key = canonical_hash(slot)
        destination = output / "rows" / (key + ".json")
        if destination.exists():
            row = read(destination)
            if row["manifest"] != identity or row["selection_slot"] != slot:
                raise ValueError("Completed row dependencies changed")
            rows.append(row)
            continue
        start_path = output / "starts" / (key + ".json")
        start = (
            read(start_path)
            if start_path.exists()
            else dict(
                epoch=time.time(),
                deadline_epoch=min(deadline - 10, time.time() + manifest["case_seconds"]),
                manifest=identity,
                selection_slot=slot,
            )
        )
        immutable(start_path, start)
        if start["manifest"] != identity:
            raise ValueError("Begun case dependencies changed")
        left = start["deadline_epoch"] - time.time()
        result = dict(status="case_budget_unavailable")
        if left > 5:
            write_artifact(
                output / "progress.json",
                dict(
                    status="running",
                    case_id=case_id,
                    completed=len(rows),
                    expected=len(ids),
                    started=start,
                    gpu_uuid=actual,
                ),
            )
            vocabulary = retrieve_vocabulary(case.problem, config=preparation.retrieval_config)
            graph = preparation.graph(case.problem, vocabulary)
            target = SemanticTargetSpec(
                canonical_hash(case.probes),
                config["desired_family_weight"],
                config["false_positive_weight"],
            )
            result = train.generated_development(
                model,
                case,
                cache,
                seconds=max(0, start["deadline_epoch"] - time.time()),
                graph=graph,
                selected_other_assignment=(
                    tuple(0 for _ in case.problem.objects)
                    if config["proposal_context"] == "selected_other_actions"
                    else None
                ),
                semantic_target=target,
                selection_slot=slot,
                stage_report_path=output / "stages" / (key + ".json"),
                post_decode_annotation_manifest=(
                    authenticate(manifest["annotation_manifest"])
                    if manifest.get("annotation_manifest")
                    else None
                ),
                annotation_directory=output / "annotation-requests",
                defer_annotation=True,
                **options,
            )
        row = dict(
            manifest=identity,
            selection_slot=slot,
            case_id=case_id,
            family=case.family,
            control=case.control,
            gpu_uuid=actual,
            result=result,
            started=start,
            finished_epoch=time.time(),
            elapsed_seconds=time.time() - start["epoch"],
        )
        immutable(destination, row)
        rows.append(row)
        # Never run additional CUDA children after an uncollected descendant.
        if "CLEANUP" in result.get("status", "").upper() or result.get("cleanup_complete") is False:
            write_artifact(
                output / "progress.json",
                dict(status="cleanup_incomplete", row=binding(destination)),
            )
            raise RuntimeError("Validation worker cleanup incomplete; quarantine this owner")
    report = dict(
        schema="exact-repair/validation-recovery-report/v1",
        status="complete",
        manifest=identity,
        expected_rows=len(ids),
        rows=[
            binding(output / "rows" / (canonical_hash(r["selection_slot"]) + ".json")) for r in rows
        ],
        technical_gate=qualification(rows, manifest["families"]),
        optimizer_updates=0,
        test_outcomes_opened=False,
        hosted_calls=0,
        gpu_uuid=actual,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    immutable(output / "report.json", report)
    write_artifact(
        output / "progress.json", dict(status="complete", report=binding(output / "report.json"))
    )
    return report


def native_probe(manifest_path, output):
    """Check original/deletion plans for fixed DEV inputs without a model or GPU."""
    from exact.repair.kernel import verify_assignment
    from exact.repair.workers import bounded_call
    from tools.repair.prepare import load_preparation

    manifest, output = read(manifest_path), Path(output)
    if manifest.get("test_allowed") is not False or manifest.get("optimizer_updates") != 0:
        raise ValueError("Native diagnostic must preserve closed TEST and saved weights")
    primary = bound(bound(manifest["weights"])["primary_manifest"])
    cases, _, _ = load_preparation(authenticate(primary["preparation"]))
    dev = {case.case_id: case for case in cases if case.split == "development"}
    rows = []
    for case_id in manifest["case_ids"]:
        case = dev[case_id]
        assignments = dict(
            original=tuple(
                next(i for i, c in enumerate(obj.candidates) if c.axioms == obj.original_axioms)
                for obj in case.problem.objects
            ),
            deletion=tuple(
                next(i for i, c in enumerate(obj.candidates) if not c.axioms)
                for obj in case.problem.objects
            ),
        )
        for kind, assignment in assignments.items():
            path = output / "rows" / (canonical_hash((case_id, kind)) + ".json")
            if path.exists():
                rows.append(binding(path))
                continue
            start = time.monotonic()
            result = bounded_call(
                verify_assignment, case.problem, assignment, timeout=10, memory_mb=8192
            )
            row = dict(
                case_id=case_id,
                family=case.family,
                control=case.control,
                kind=kind,
                assignment=assignment,
                status=result.status,
                detail=result.detail,
                cleanup_complete=result.cleanup_complete,
                elapsed_seconds=time.monotonic() - start,
                verification=result.value.to_dict() if result.status == "complete" else None,
                verified=result.status == "complete" and result.value.authorizes,
            )
            immutable(path, row)
            rows.append(binding(path))
    immutable(
        output / "report.json",
        dict(
            status="complete",
            manifest=binding(Path(manifest_path)),
            rows=rows,
            hosted_calls=0,
            optimizer_updates=0,
            test_opened=False,
        ),
    )


def aggregate(manifest_path, report_paths, output):
    """Join disjoint GPU shards; paired conditions use the same hardware per case."""
    manifest = read(manifest_path)
    identity = binding(Path(manifest_path))
    refs, rows = [], []
    for path in report_paths:
        report = read(path)
        if (
            report["manifest"] != identity
            or report["status"] != "complete"
            or report["expected_rows"] != len(report["rows"])
            or report["test_outcomes_opened"] is not False
            or report["optimizer_updates"] != 0
            or report["hosted_calls"] != 0
        ):
            raise ValueError("Validation shard identity/status changed")
        for ref in report["rows"]:
            row = bound(ref)
            if row["manifest"] != identity:
                raise ValueError("Validation row manifest changed")
            expected = manifest["case_gpu_uuids"][row["case_id"]]
            if (
                row["gpu_uuid"].removeprefix("GPU-") != expected.removeprefix("GPU-")
                or report["gpu_uuid"] != row["gpu_uuid"]
                or row["selection_slot"]["case_id"] != row["case_id"]
            ):
                raise ValueError("Paired case hardware assignment changed")
            refs.append(ref)
            rows.append(row)
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(manifest["case_ids"]):
        raise ValueError("Validation shard coverage is overlapping or incomplete")
    immutable(
        Path(output) / "report.json",
        dict(
            schema="exact-repair/validation-recovery-report/v1",
            status="complete",
            manifest=identity,
            expected_rows=len(ids),
            rows=refs,
            shard_reports=[binding(Path(p)) for p in report_paths],
            technical_gate=qualification(rows, manifest["families"]),
            optimizer_updates=0,
            test_outcomes_opened=False,
            hosted_calls=0,
        ),
    )


def annotate_report(report_path, output):
    """CPU-only completion of saved DEV comparisons; never regenerate a repair."""
    from exact.repair.api import write_artifact
    from exact.repair.protocol import load_protocol_v3, training_projection_v3
    from exact.repair.semantic_fidelity import offline_plan_label, read_fidelity_training_artifact
    from tools.repair.corrective_semantics import annotate_decoded
    from tools.repair.prepare import case_from_dict
    from exact.repair.learning import generated_checkpoint_criterion

    report, output = read(report_path), Path(output)
    manifest = bound(report["manifest"])
    recovery = bound(manifest["weights"])
    primary = bound(recovery["primary_manifest"])
    config = runtime_configuration(
        training_projection_v3(
            load_protocol_v3(authenticate(primary["protocol"]), for_execution=True)
        )
    )
    annotation_manifest = authenticate(manifest["annotation_manifest"])
    if manifest.get("test_allowed") is not False or report["expected_rows"] != 32:
        raise ValueError("Primary annotation requires the full fixed DEV denominator")
    declaration = read(annotation_manifest)
    expected = manifest["case_ids"]
    source_rows = [bound(reference) for reference in report["rows"]]
    observed = [row["case_id"] for row in source_rows]
    if (
        len(observed) != 32
        or len(observed) != len(set(observed))
        or set(observed) != set(expected)
        or any(row["manifest"] != report["manifest"] for row in source_rows)
    ):
        raise ValueError("Decoded DEV coverage or provenance changed")
    immutable(output / "source-binding.json", binding(Path(report_path)))
    historical = bound(manifest["historical_review"])
    predecessor = next(
        m
        for m in historical["models"]
        if m["seed"] == recovery["seed"] and m["condition"] == recovery["condition"]
    )
    if any(r.get("selection_criterion") is not None for r in predecessor["development"]):
        raise ValueError("A prior eligible checkpoint must participate in unchanged selection")
    semantic_ids = [
        case_id
        for case_id in expected
        if canonical_hash(
            dict(
                seed=recovery["seed"],
                supervision_condition=recovery["condition"],
                epoch=recovery["epoch"],
                case_id=case_id,
            )
        )
        in declaration["frozen_annotation_slots"]
    ]
    results = []
    deadline = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"])
    for reference, source in zip(report["rows"], source_rows):
        destination = output / "rows" / (canonical_hash(source["selection_slot"]) + ".json")
        if destination.exists():
            row = read(destination)
            if row["source"] != reference:
                raise ValueError("Annotation resume changed its decoded plan")
            results.append(row["result"])
            continue
        result = source["result"]
        decoded = result.get("decoded", {})
        if decoded.get("annotation_pending") and decoded.get("annotation_request"):
            source_request = Path(decoded["annotation_request"])
            request = read(source_request)
            request_path = (
                Path(manifest["annotation_state_directory"])
                / request["request_identity"]
                / "request.json"
            )
            immutable(request_path, request)
            immutable(
                request_path.with_name("origin.json"),
                dict(source=binding(source_request), decoded_row=reference),
            )
            if (
                request["selection_slot"] != source["selection_slot"]
                or request["model_hash"] != recovery["artifact"]["model_hash"]
            ):
                raise ValueError("Annotation request changed model/slot identity")
            left = max(0, deadline - time.time() - 10)
            artifact = (
                annotate_decoded(request_path, annotation_manifest, seconds=left)
                if left > 92
                else None
            )
            # A deferred transport remains resumable; terminal invalid/abstained
            # observations remain unavailable and are never sampled again.
            waits = [read(p) for p in (request_path.parent / "annotation").glob("*/receipt.json")]
            pending = any(r.get("status") == "rate_limit_deferred" for r in waits)
            if pending or left <= 92:
                deferred = dict(
                    status="transport_deferred",
                    reason="confirmed_provider_backoff" if pending else "worker_slice_exhausted",
                    request=binding(request_path),
                    source=reference,
                    costs_reset=False,
                    completed=len(results),
                )
                write_artifact(output / "progress.json", deferred)
                write_artifact(output / "report.json", deferred)
                return deferred
            case = case_from_dict(request["case"])
            labels = (
                read_fidelity_training_artifact(Path(artifact), "development") if artifact else {}
            )
            weak = offline_plan_label(
                case,
                tuple(request["assignment"]),
                tuple(map(tuple, request["profile"])),
                labels.get(case.case_id, ()),
                role="evaluator",
                use_policy=config["development_use_policy"],
                rating_aggregation=config["plan_rating_aggregation"],
            )
            score = weak.benefit if weak is not None else None
            symbolic = request["symbolic_label"]
            mixture = config["mixture_symbolic_weight"]
            benefit = (
                mixture * symbolic["benefit"] + (1 - mixture) * score
                if symbolic["benefit"] is not None and score is not None
                else None
            )
            decoded.update(
                status="verified" if benefit is not None else "unknown_selected_labels",
                selected_utility=benefit - symbolic["cost"] if benefit is not None else None,
                semantic_benefit=benefit,
                weak_benefit=score,
                symbolic_benefit=symbolic["benefit"],
                annotation_artifact=str(artifact) if artifact else None,
                annotation_pending=False,
            )
        result = dict(result, case_id=source["case_id"])
        immutable(destination, dict(source=reference, result=result))
        results.append(result)
    criterion = generated_checkpoint_criterion(
        results,
        expected_case_ids=expected,
        semantic_case_ids=semantic_ids,
        minimum_coverage=config["minimum_generated_coverage"],
        uncertainty_z=config["quality_uncertainty_z"],
        fallback=config["missing_label_fallback"],
    )
    final = dict(
        schema="exact-repair/recovered-dev-selection/v1",
        status="complete",
        source=binding(Path(report_path)),
        rows=len(results),
        selection_criterion=criterion,
        selected_checkpoint=recovery["artifact"] if criterion is not None else None,
        epoch=recovery["epoch"],
        selection_rule_unchanged=True,
        prior_epoch5_outcomes_preserved=True,
        test_opened=False,
        costs_reset=False,
    )
    final_path = output / "report.json"
    if final_path.exists() and read(final_path).get("status") == "complete":
        immutable(final_path, final)
    else:
        write_artifact(final_path, final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--native-probe", action="store_true")
    parser.add_argument("--annotate-report", action="store_true")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--aggregate", nargs="+")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.manifest, args.aggregate, args.output)
    elif args.annotate_report:
        annotate_report(args.manifest, args.output)
    elif args.native_probe:
        native_probe(args.manifest, args.output)
    else:
        run(args.manifest, args.output, gpu_uuid=args.gpu_uuid)


if __name__ == "__main__":
    main()
