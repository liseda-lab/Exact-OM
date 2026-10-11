"""Re-evaluate authenticated saved weights; no optimizer or TEST entry point.

Original training states and DEV rows are never changed. The manifest binds each
recovery's code, weights, cases and budget. Completed rows are reusable only under
that same identity; begun cases retain their original deadlines after a restart.
"""

from __future__ import annotations

import argparse
import json
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


def run(manifest_path, output):
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
    actual = owned_device(manifest["gpu_uuid"])
    identity = binding(Path(manifest_path))
    immutable(output / "manifest-binding.json", identity)
    recovery = bound(manifest["weights"])
    primary = bound(recovery["primary_manifest"])
    protocol = training_projection_v3(
        load_protocol_v3(authenticate(primary["protocol"]), for_execution=True)
    )
    cases, caches, _ = load_preparation(authenticate(primary["preparation"]))
    config = train._protocol_arguments(protocol)
    config["profile"] = train._protocol_profile(protocol)
    dev = {
        case.case_id: (case, cache)
        for case, cache in scheduled_runtime_development(cases, caches, config)
    }
    ids = manifest["case_ids"]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
