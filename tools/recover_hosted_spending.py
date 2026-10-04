"""Continue a verified E25 inference checkpoint after an accounting-only amendment."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

_ALLOWED = {
    "exact/experiments/budget.py",
    "exact/experiments/campaign.py",
    "exact/experiments/runtime.py",
    "exact/llm/ledger.py",
    "exact/utils/hosted_spending.py",
}


def verify_code(old_code, code, repair):
    from exact.experiments.runtime import _code_identity

    if (
        repair.get("migration") != "hosted-spending-checkpoint-v1"
        or repair.get("scientific_choices_unchanged") is not True
        or repair.get("reporting_labels_exposed") is not False
        or set(repair.get("changes", {})) != _ALLOWED
    ):
        raise ValueError("Expected the explicitly bound accounting-only amendment")
    before = _code_identity(old_code, evaluation=False)
    for evaluation in (False, True):
        old = dict(_code_identity(old_code, evaluation=evaluation)["files"])
        new = dict(_code_identity(code, evaluation=evaluation)["files"])
        for name, change in repair["changes"].items():
            if name not in old and name not in new:
                continue
            if old.pop(name, None) != change["before"] or new.pop(name, None) != change["after"]:
                raise ValueError("Accounting amendment source hashes differ")
        if old != new:
            raise ValueError("Scientific implementation changed outside the accounting amendment")
    return before


def verify_identity(saved, expected, old_impl):
    target = copy.deepcopy(expected)
    target.pop("artifact_id")
    target["implementation"] = old_impl
    source = dict(saved)
    source.pop("artifact_id")
    if source != target:
        raise ValueError("Checkpoint input/config/role/seed identity differs")


def relocate_inference_checkpoint(payload, source_cell, target_cell, routing_binding):
    """Rebind only gate paths after verifying their original numerical bytes."""
    from exact.impl.trainer.checkpointing import CheckpointingMixin

    fingerprint = CheckpointingMixin._hash_checkpoint_fingerprint_payload
    source_cell, target_cell = Path(source_cell).resolve(), Path(target_cell).resolve()
    original = copy.deepcopy(payload)
    models = original.get("checkpoint_fingerprint_payload", {}).get("models", [])
    if len(models) != 1 or models[0].get("class") != "PairAdaptiveSemanticScorer":
        raise ValueError("Expected the original E25 primary scorer fingerprint")
    model = models[0]
    if model.get("fingerprint") != fingerprint(model["payload"]) or original.get(
        "checkpoint_fingerprint"
    ) != fingerprint(original["checkpoint_fingerprint_payload"]):
        raise ValueError("Original trainer checkpoint fingerprint is inconsistent")
    channels = model["payload"]["pair_adaptive_channels"]["experiments"]
    gate = channels["llm"]["gate"]
    provenance = channels["gate_artifact"]
    old_gate = Path(gate["artifact"])
    if (
        gate.get("mode") != "forced_sample"
        or str(old_gate) != provenance.get("path")
        or not old_gate.is_absolute()
        or not old_gate.resolve().is_relative_to(source_cell / "routing")
        or old_gate.name != "selection.json"
        or old_gate.parent.parent != source_cell / "routing"
    ):
        raise ValueError("Expected the original cell's forced-source routing artifact")
    manifest = read(verified(routing_binding))
    if manifest.get("schema_version") != 1 or manifest.get("directory") != str(old_gate.parent):
        raise ValueError("Routing manifest does not bind the checkpoint's gate directory")
    files = manifest.get("files", {})
    inventory = {
        path.relative_to(old_gate.parent).as_posix()
        for path in old_gate.parent.rglob("*")
        if path.is_file()
    }
    if not files or set(files) != inventory:
        raise ValueError("Routing manifest must bind every retained prepass file")
    outputs = {}
    for relative, item in files.items():
        path = verified(item)
        if path != old_gate.parent / relative or not path.resolve().is_relative_to(old_gate.parent):
            raise ValueError("Routing binding escapes the original gate directory")
        # This file contains its old absolute selection path. The regular cached
        # prepass regenerates this provenance from the retained numerical shards.
        if relative != "prepass.json":
            outputs[path.relative_to(source_cell).as_posix()] = path
    if (
        files["selection.json"]["sha256"] != provenance.get("sha256")
        or old_gate.stat().st_size != provenance.get("bytes")
        or read(old_gate).get("mode") != "forced_sample"
    ):
        raise ValueError("Retained routing selection differs from checkpoint gate provenance")
    new_gate = target_cell / old_gate.relative_to(source_cell)
    channels["llm"]["gate"]["artifact"] = str(new_gate)
    channels["gate_artifact"]["path"] = str(new_gate)
    model["fingerprint"] = fingerprint(model["payload"])
    original["checkpoint_fingerprint"] = fingerprint(original["checkpoint_fingerprint_payload"])

    # Prove the only changes are the two relocated paths and their dependent
    # fingerprints; inference mappings, timings and all model choices stay intact.
    restored = copy.deepcopy(original)
    restored_model = restored["checkpoint_fingerprint_payload"]["models"][0]
    restored_channels = restored_model["payload"]["pair_adaptive_channels"]["experiments"]
    restored_channels["llm"]["gate"]["artifact"] = str(old_gate)
    restored_channels["gate_artifact"]["path"] = str(old_gate)
    restored_model["fingerprint"] = payload["checkpoint_fingerprint_payload"]["models"][0][
        "fingerprint"
    ]
    restored["checkpoint_fingerprint"] = payload["checkpoint_fingerprint"]
    if restored != payload:
        raise ValueError("Checkpoint relocation changed content outside path provenance")
    return (
        original,
        outputs,
        {
            "source_gate": files["selection.json"],
            "replacement_gate_path": str(new_gate),
            "routing_manifest": routing_binding,
            "routing_files_restored": len(outputs),
            "routing_prepass_metadata_regenerated": "prepass.json" in files,
            "original_trainer_fingerprint": payload["checkpoint_fingerprint"],
            "replacement_trainer_fingerprint": original["checkpoint_fingerprint"],
            "changed_checkpoint_fields": [
                "checkpoint_fingerprint_payload.models[0].payload.pair_adaptive_channels.experiments.llm.gate.artifact",
                "checkpoint_fingerprint_payload.models[0].payload.pair_adaptive_channels.experiments.gate_artifact.path",
                "checkpoint_fingerprint_payload.models[0].fingerprint",
                "checkpoint_fingerprint",
            ],
        },
    )


def _cells(recipe, campaign, runtime):
    from exact.experiments import harness
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        materialize_campaign,
    )

    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, runtime / "declarations/screen", stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == recipe["scientific_step"])
    experiments, manifests = {}, []
    for identifier in source.config.depends_on:
        step = next(s for s in lock.steps if s.id == identifier)
        if step.external_selection is None:
            raise ValueError("Checkpoint continuation requires completed producer bindings")
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
    if (
        recipe["scientific_step"] != "E25-forced"
        or len(cells) != 1
        or cells[0].arm_id != "forced_sources"
        or cells[0].task_id != "D0_E03-global_alignment"
        or cells[0].seed != 17
    ):
        raise ValueError("Only the original E25 forced diagnostic may use this migration")
    return suite, cells[0]


def import_saved(recipe, campaign, runtime, code, *, verify_only=False):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery

    settings = recipe["hosted_spending_repair"]
    completion = read(verified(settings["source_completion"]))
    if (
        completion.get("status") != "failed"
        or completion.get("exit_code") != 1
        or verified(settings["source_exit"]).read_text().strip() != "1"
    ):
        raise ValueError("Checkpoint source must be a terminal failed attempt")
    source_root = Path(settings["source_runtime"])
    repair = read(verified(settings["repair_record"]))
    old_impl = verify_code(Path(settings["source_code"]), code, repair)
    suite, cell = _cells(recipe, campaign, runtime)
    manifest = verified(settings["source_manifest"])
    expected_path = (
        source_root
        / "screen/runs"
        / cell.experiment_id
        / cell.arm_id
        / cell.task_id
        / f"seed-{cell.seed}"
        / "experiment_manifest.json"
    )
    report = read(manifest)
    if (
        manifest != expected_path
        or report.get("status") != "failed"
        or report.get("extraction_complete") is not False
        or report.get("resolved_config_hash") != cell.config_hash
        or load_yaml_mapping(manifest.parent / "_inputs/resolved.config.yaml")
        != cell.resolved_config
    ):
        raise ValueError("Checkpoint source configuration or completion differs")
    recovery = CellRecovery(
        replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True}),
        harness._provenance_payload(cell, suite, workdir=code),
        code,
    )
    saved_path = verified(settings["source_checkpoint"])
    payload = read(saved_path)
    source = ArtifactStore(source_root)
    artifact_id = ArtifactStore._identity(payload["identity"])
    if (
        saved_path
        != source.directory / "checkpoints" / artifact_id / f"{payload['sequence']:08d}.json"
    ):
        raise ValueError("Checkpoint binding is outside its source artifact")
    if source.latest_checkpoint(artifact_id) != payload:
        raise ValueError("Bound checkpoint is not the latest verified saved boundary")
    verify_identity(payload["identity"], recovery.identities["extraction"], old_impl)
    parent = source.verify(recovery.identities["inputs"]["artifact_id"])
    if parent["identity"] != recovery.identities["inputs"]:
        raise ValueError("Checkpoint inputs changed")
    count = len(payload["completed_ids"])
    if (
        not 0 < count < payload["cursor"]["dataset_rows"]
        or payload["cursor"]["next_pair"] != count
        or len(set(payload["completed_ids"])) != count
        or not any(name.startswith("checkpoints/inference_") for name in payload["outputs"])
    ):
        raise ValueError("Expected an exact, incomplete inference checkpoint boundary")
    store = ArtifactStore(runtime)
    identity = recovery.identities["extraction"]
    outputs = {name: source._blob(item["sha256"]) for name, item in payload["outputs"].items()}
    inference = [name for name in outputs if name.startswith("checkpoints/inference_")]
    if len(inference) != 1:
        raise ValueError("Expected one exact saved trainer inference checkpoint")
    relocated, routing_outputs, relocation = relocate_inference_checkpoint(
        read(outputs[inference[0]]),
        manifest.parent,
        cell.output_dir,
        settings["source_routing_manifest"],
    )
    if relocated.get("processed_examples") != count:
        raise ValueError("Trainer checkpoint count differs from the retained boundary")
    outputs[inference[0]] = (json.dumps(relocated, indent=2, ensure_ascii=False) + "\n").encode()
    if set(outputs) & set(routing_outputs):
        raise ValueError("Routing migration would replace an existing checkpoint output")
    outputs.update(routing_outputs)
    if not verify_only:
        previous = store.latest_checkpoint(identity["artifact_id"])
        if previous is None:
            store.import_artifact(source_root, parent["identity"]["artifact_id"])
            store.checkpoint(
                identity,
                completed_ids=payload["completed_ids"],
                cursor=payload["cursor"],
                state=payload["state"],
                outputs=outputs,
            )
        elif previous["cursor"]["next_pair"] < count:
            raise ValueError("Destination checkpoint regressed behind the imported boundary")
    result = {
        "status": "pass",
        "verify_only": verify_only,
        "completed_pairs": count,
        "total_pairs": payload["cursor"]["dataset_rows"],
        "source_checkpoint": binding(saved_path),
        "repair_record": settings["repair_record"],
        "original_artifact_id": artifact_id,
        "replacement_artifact_id": identity["artifact_id"],
        "numerical_output_bytes_unchanged": True,
        "checkpoint_metadata_relocated": relocation,
        "accounting_handoff": "Caller must retain the latest closed cumulative account before import",
    }
    write(
        runtime
        / (
            "spending-checkpoint-verification.json"
            if verify_only
            else "spending-checkpoint-import.json"
        ),
        result,
    )
    return result
