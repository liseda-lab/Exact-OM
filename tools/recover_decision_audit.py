"""Resume completed E25 inference after an output-only audit serialization repair."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write


def cells_for_repair(recipe, campaign, runtime):
    from exact.experiments import harness
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        materialize_campaign,
    )

    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, runtime / "declarations/screen", stage="screen")
    # Match execute_campaign: generated policy paths must be the worker runtime,
    # not materialize_campaign's standalone declaration directory.
    suite = replace(suite, campaign={**(suite.campaign or {}), "root": str(runtime)})
    source = next(s for s in suite.sources if s.config.experiment_id == recipe["scientific_step"])
    experiments, manifests = {}, []
    required = set(source.config.depends_on) | set(
        source.config.frozen_constants.get("campaign_v2", {}).get("requires", [])
    )
    for identifier in sorted(required):
        step = next(s for s in lock.steps if s.id == identifier)
        if step.external_selection is None:
            raise ValueError("Audit recovery requires verified completed producers")
        historical = external_selection_result(
            lock, step, campaign.parent, producer_manifests=manifests
        )
        producer = next(s for s in suite.sources if s.config.experiment_id == identifier)
        experiments[identifier] = harness._bind_external_selection(producer, suite, historical)
    inherited = harness.inherited_selection_overlay(
        {"experiments": experiments}, source.config.depends_on
    )
    source, inherited = harness._materialize_campaign_evidence(
        source, suite, [], experiments, inherited, external_manifests=manifests
    )
    cells = harness.build_cells(
        suite, source, stage="screen", output_root=runtime.parent, inherited_overlay=inherited
    )
    expected_arms = {
        "E25-oracles": {"decision_off", "oracle_observed", "oracle_perfect"},
        "E25-trust": {"trust_shipped", "trust_constant"},
    }.get(recipe["scientific_step"])
    if (
        expected_arms is None
        or {c.arm_id for c in cells} != expected_arms
        or len(cells) != len(expected_arms)
    ):
        raise ValueError("Audit recovery requires the complete original E25 comparison")
    if any(
        c.task_id != "D0_E03-global_alignment" or c.seed != 17 or c.source_cap != 300 for c in cells
    ):
        raise ValueError("Audit recovery population changed")
    return suite, cells


def relocate(payload, new_gate, *, additional=False):
    """Relocate only verified identical gate bytes and dependent fingerprint metadata."""
    from exact.impl.trainer.checkpointing import CheckpointingMixin

    digest = CheckpointingMixin._hash_checkpoint_fingerprint_payload
    result = copy.deepcopy(payload)
    key = "fingerprint_payload" if additional else "checkpoint_fingerprint_payload"
    model = result[key]["models"][0]
    if model["class"] != "PairAdaptiveSemanticScorer" or model["fingerprint"] != digest(
        model["payload"]
    ):
        raise ValueError("Original model fingerprint changed")
    if not additional and result["checkpoint_fingerprint"] != digest(result[key]):
        raise ValueError("Original inference fingerprint changed")
    channels = model["payload"]["pair_adaptive_channels"]["experiments"]
    gate, provenance = channels["llm"]["gate"], channels["gate_artifact"]
    old_gate = Path(gate["artifact"])
    if (
        gate["mode"] not in {"oracle_replay", "oracle_perfect"}
        or provenance["path"] != str(old_gate)
        or binding(old_gate)["sha256"] != provenance["sha256"]
        or old_gate.stat().st_size != provenance["bytes"]
    ):
        raise ValueError("Original replay gate bytes changed")
    if (
        binding(new_gate)["sha256"] != provenance["sha256"]
        or Path(new_gate).stat().st_size != provenance["bytes"]
    ):
        raise ValueError("Recovery replay gate differs numerically")
    gate["artifact"] = str(new_gate)
    provenance["path"] = str(new_gate)
    model["fingerprint"] = digest(model["payload"])
    if additional:
        result["candidate_records_path"] = (
            "inference_additional_models_" + digest(result[key])[:12] + ".jsonl.zst"
        )
    else:
        result["checkpoint_fingerprint"] = digest(result[key])
    # An inverse transformation must reconstruct every original field exactly.
    inverse = copy.deepcopy(result)
    m = inverse[key]["models"][0]
    c = m["payload"]["pair_adaptive_channels"]["experiments"]
    c["llm"]["gate"]["artifact"] = str(old_gate)
    c["gate_artifact"]["path"] = str(old_gate)
    m["fingerprint"] = payload[key]["models"][0]["fingerprint"]
    if additional:
        inverse["candidate_records_path"] = payload["candidate_records_path"]
    else:
        inverse["checkpoint_fingerprint"] = payload["checkpoint_fingerprint"]
    if inverse != payload:
        raise ValueError("Checkpoint relocation changed numerical content")
    return result


def import_saved(recipe, campaign, runtime, code, *, verify_only=False):
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _code_identity
    from tools.validation_resume import _audit_repair_compatibility

    settings = recipe["decision_audit_repair"]
    completion = read(verified(settings["source_completion"]))
    if (
        completion.get("status") != "failed"
        or completion.get("exit_code") != 1
        or verified(settings["source_exit"]).read_text().strip() != "1"
    ):
        raise ValueError("Audit source must be a terminal failed attempt")
    source_root = Path(settings["source_runtime"])
    source = ArtifactStore(source_root)
    suite, cells = cells_for_repair(recipe, campaign, runtime)
    reports = []
    for cell in cells:
        record = settings["cells"][cell.arm_id]
        manifest_path = verified(record["manifest"])
        manifest = read(manifest_path)
        expected = (
            source_root
            / "screen/runs"
            / cell.experiment_id
            / cell.arm_id
            / cell.task_id
            / f"seed-{cell.seed}"
            / "experiment_manifest.json"
        )
        if (
            manifest_path != expected
            or manifest["status"] != "failed"
            or manifest.get("extraction_complete") is not False
        ):
            raise ValueError("Original cell completion or provenance changed")
        checkpoint_path = verified(record["checkpoint"])
        checkpoint = read(checkpoint_path)
        original = checkpoint["identity"]
        if (
            checkpoint_path
            != source.directory
            / "checkpoints"
            / original["artifact_id"]
            / f"{checkpoint['sequence']:08d}.json"
            or source.latest_checkpoint(original["artifact_id"]) != checkpoint
        ):
            raise ValueError("Bound checkpoint is not the latest verified boundary")
        recovery = CellRecovery(
            replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True}),
            harness._provenance_payload(cell, suite, workdir=code),
            code,
        )
        current = recovery.identities["extraction"]
        omit = {"artifact_id", "implementation"}
        if {k: v for k, v in original.items() if k not in omit} != {
            k: v for k, v in current.items() if k not in omit
        }:
            raise ValueError(
                "Audit recovery input/config/role/seed/package identity differs: " + cell.arm_id
            )
        if current["implementation"] != _code_identity(code, evaluation=False):
            raise ValueError("Current executable source differs")
        _audit_repair_compatibility(
            original["implementation"], current["implementation"], settings["original_revision"]
        )
        count = checkpoint["cursor"].get("dataset_rows")
        if (
            count != 5842
            or checkpoint["cursor"].get("next_pair") != count
            or len(set(checkpoint["completed_ids"])) != count
        ):
            raise ValueError("Expected the complete original 5842-pair checkpoint")
        outputs = {
            name: source._blob(item["sha256"]) for name, item in checkpoint["outputs"].items()
        }
        primary = [
            name
            for name in outputs
            if name.startswith("checkpoints/inference_")
            and "additional_models" not in name
            and name.endswith(".json")
        ]
        extra = [
            name
            for name in outputs
            if name.startswith("checkpoints/inference_additional_models_")
            and name.endswith(".json")
        ]
        if len(primary) != 1 or len(extra) != 1:
            raise ValueError("Expected one primary and one selector checkpoint")
        new_gate = Path(cell.resolved_config["llm"]["experiment"]["gate"]["artifact"])
        inference = read(outputs[primary[0]])
        selector = read(outputs[extra[0]])
        if (
            inference.get("processed_examples") != count
            or selector.get("complete") is not True
            or selector.get("candidate_records_count") != count
        ):
            raise ValueError("Inner inference or selector checkpoint is incomplete")
        relocated = relocate(inference, new_gate)
        relocated_selector = relocate(selector, new_gate, additional=True)
        outputs[primary[0]] = (json.dumps(relocated, indent=2) + "\n").encode()
        old_rows = str(Path(extra[0]).parent / selector["candidate_records_path"])
        new_rows = str(Path(extra[0]).parent / relocated_selector["candidate_records_path"])
        rows = outputs.pop(old_rows)
        outputs.pop(extra[0])
        new_meta = new_rows.removesuffix("l.zst")
        outputs[new_rows] = rows
        outputs[new_meta] = (json.dumps(relocated_selector, indent=2) + "\n").encode()
        if not verify_only:
            store = ArtifactStore(runtime)
            if store.latest_checkpoint(current["artifact_id"]) is not None:
                raise ValueError("Recovery destination already owns a checkpoint")
            for parent in original["parents"]:
                store.import_artifact(source_root, parent)
            store.checkpoint(
                current,
                completed_ids=checkpoint["completed_ids"],
                cursor=checkpoint["cursor"],
                state=checkpoint["state"],
                outputs=outputs,
            )
        reports.append(
            {
                "arm": cell.arm_id,
                "completed_pairs": count,
                "original_checkpoint": record["checkpoint"],
                "original_artifact_id": original["artifact_id"],
                "replacement_artifact_id": current["artifact_id"],
                "selector_rows_unchanged": True,
                "new_gate": binding(new_gate),
            }
        )
    result = {
        "status": "passed",
        "verify_only": verify_only,
        "cells": reports,
        "scientific_choices_unchanged": True,
        "new_hosted_requests": 0,
    }
    write(
        runtime
        / ("audit-recovery-verification.json" if verify_only else "audit-recovery-import.json"),
        result,
    )
    return result
