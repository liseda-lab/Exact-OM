"""Strict E07 artifact migration across reviewed ledger/retry bookkeeping changes."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

FILES = {"exact/llm/ledger.py", "exact/llm/routing.py"}
ARMS = {
    "brief_256_binary",
    "brief_binary",
    "facts_binary",
    "retrieved_listwise",
    "packet_binary",
    "facts_listwise",
}
STAGES = ("inputs", "extraction", "evaluation")


def verify_code(old_code, code, repair):
    from exact.experiments.runtime import _code_identity

    old = _code_identity(old_code, evaluation=False)
    new = _code_identity(code, evaluation=False)
    changes = repair.get("changes", {})
    if (
        repair.get("migration") != "e07-ledger-retry-v1"
        or not {"exact/llm/ledger.py"} <= set(changes) <= FILES
    ):
        raise ValueError("Only the pinned ledger/retry bookkeeping repair may migrate E07")
    before, after = dict(old["files"]), dict(new["files"])
    for name, change in changes.items():
        if before.pop(name, None) != change["before"] or after.pop(name, None) != change["after"]:
            raise ValueError("Reviewed hosted synchronization hashes differ")
    if before != after or _code_identity(old_code, evaluation=True) != _code_identity(
        code, evaluation=True
    ):
        raise ValueError("Code changed outside reviewed hosted synchronization")
    if (
        repair.get("scientific_choices_unchanged") is not True
        or repair.get("reporting_labels_exposed") is not False
    ):
        raise ValueError("Migration requires unchanged development science")
    return old


def verify_identity(saved, expected, old_impl, *, stage, old_extraction=None):
    value = copy.deepcopy(expected)
    value.pop("artifact_id")
    if stage == "extraction":
        value["implementation"] = old_impl
    elif stage == "evaluation":
        value["parents"] = [old_extraction]
    actual = dict(saved)
    actual.pop("artifact_id")
    if actual != value:
        raise ValueError("Artifact identity differs beyond hosted synchronization: " + stage)


def verify_checkpoint(checkpoint, store, config):
    """Check the complete ordered inference prefix and its matching runner cursor."""
    outputs, cursor = checkpoint["outputs"], checkpoint["cursor"]
    count, total = cursor["next_pair"], cursor["dataset_rows"]
    if count <= 0 or count > total or len(set(checkpoint["completed_ids"])) != count:
        raise ValueError("E07 checkpoint cursor or unique membership differs")
    with store._blob(outputs["dataset/dataset.csv"]["sha256"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row.get("inference") not in {"True", "False"} for row in rows):
        raise ValueError("E07 checkpoint lacks explicit inference membership")
    experiment = config.get("llm", {}).get("experiment", {})
    gate = experiment.get("gate", {})
    if (
        experiment.get("enabled") is not True
        or gate.get("mode") != "source_top_fraction"
        or gate.get("artifact") is not None
    ):
        raise ValueError("E07 checkpoint requires its original fitted population gate")
    # alignment.py creates fitting_gate_config for this original gate. Runner.run
    # then sorts all six E07 arms for grouped decisions, after dataset.csv is saved.
    columns = ("Src", "SrcKind", "Tgt", "TgtKind")
    active = sorted(
        (row for row in rows if row["inference"] == "True"),
        key=lambda row: tuple(row[k] for k in columns),
    )
    ordered = [json.dumps([row[k] for k in columns]) for row in active]
    if len(ordered) != total or checkpoint["completed_ids"] != ordered[:count]:
        raise ValueError("E07 checkpoint is not the exact ordered inference prefix")
    names = [name for name in outputs if name.startswith("checkpoints/inference_")]
    if len(names) != 1:
        raise ValueError("E07 checkpoint must bind one inference cursor")
    saved = read(store._blob(outputs[names[0]]["sha256"]))
    expected = dict(
        kind="inference",
        total_examples=total,
        processed_examples=count,
        mappings_count=count,
        results_json_count=count,
        explanation_records_count=count,
        explanation_index_path="../explanations/index.json",
    )
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("E07 runner cursor or explanation coverage differs")


def cells_for(recipe, campaign, runtime):
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
            raise ValueError("Recovery requires verified completed producers")
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
    if recipe["scientific_step"] != "E07" or len(cells) != 6 or {c.arm_id for c in cells} != ARMS:
        raise ValueError("Migration requires all six original E07 arms")
    return suite, cells


def import_saved(recipe, campaign, runtime, code, *, verify_only=False):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _hash, _verify_materialized

    settings = recipe["e07_ledger_repair"]
    old_root = Path(settings["source_runtime"])
    repair = read(verified(settings["repair_record"]))
    old_impl = verify_code(Path(settings["source_code"]), code, repair)
    suite, cells = cells_for(recipe, campaign, runtime)
    old_store, store = ArtifactStore(old_root), ArtifactStore(runtime)
    imports, rows = [], []
    for cell in cells:
        path = verified(settings["manifests"][cell.arm_id])
        if (
            path
            != old_root
            / "screen/runs"
            / cell.experiment_id
            / cell.arm_id
            / cell.task_id
            / f"seed-{cell.seed}"
            / "experiment_manifest.json"
        ):
            raise ValueError("Source cell outside declared runtime")
        report = read(path)
        expected = dict(
            experiment_id=cell.experiment_id,
            arm_id=cell.arm_id,
            task_id=cell.task_id,
            stage="screen",
            seed=cell.seed,
            split_role=cell.split_role,
            reference_role=cell.reference_role,
            source_cap=cell.source_cap,
            arm_role=cell.arm_role,
            resolved_config_hash=cell.config_hash,
            generate_rationales=False,
            resolved_supervision=cell.resolved_supervision,
        )
        if (
            any(report.get(k) != v for k, v in expected.items())
            or load_yaml_mapping(path.parent / "_inputs/resolved.config.yaml")
            != cell.resolved_config
        ):
            raise ValueError("Original config, population, seed or role differs")
        recovery = CellRecovery(
            replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True}),
            harness._provenance_payload(cell, suite, workdir=code),
            code,
        )
        identities = recovery.identities
        if cell.arm_id == "brief_256_binary":
            if (
                report.get("status") != "complete"
                or report.get("return_code") != 0
                or report.get("extraction_complete") is not True
            ):
                raise ValueError("Control is not complete")
            artifacts = report["recovery"]["artifacts"]
            if set(artifacts) != set(STAGES):
                raise ValueError("Control lacks complete artifact provenance")
            saved = {}
            for stage in STAGES:
                payload = old_store.verify(artifacts[stage])
                verify_identity(
                    payload["identity"],
                    identities[stage],
                    old_impl,
                    stage=stage,
                    old_extraction=artifacts["extraction"],
                )
                if stage != "inputs":
                    _verify_materialized(old_store, payload, path.parent)
                saved[stage] = payload
            m = saved["extraction"]["outputs"]["stats/execution_measurement.json"]
            measurement = read(old_store._blob(m["sha256"]))
            if (
                measurement.get("artifact_id") != artifacts["extraction"]
                or measurement.get("schema_version") != 1
            ):
                raise ValueError("Original execution measurement mismatch")
            measurement = {
                **measurement,
                "artifact_id": identities["extraction"]["artifact_id"],
                "identity_migration": {
                    "reason": "e07-ledger-retry-v1",
                    "source_artifact_id": artifacts["extraction"],
                    "source_measurement_sha256": m["sha256"],
                    "previous": measurement.get("identity_migration"),
                },
            }
            imports.append((cell, identities, path, report, saved, measurement))
            rows.append({"arm": cell.arm_id, "action": "reuse_complete", "artifacts": artifacts})
        else:
            if report.get("status") != "failed" or report.get("extraction_complete") is not False:
                raise ValueError("Expected incomplete treatment")
            source_identity = read(path.parent / "recovery-runtime.json")["identity"]
            verify_identity(source_identity, identities["extraction"], old_impl, stage="extraction")
            checkpoint = old_store.latest_checkpoint(source_identity["artifact_id"])
            count = 0
            if checkpoint is not None:
                if checkpoint["identity"] != source_identity:
                    raise ValueError("Treatment checkpoint identity differs")
                verify_checkpoint(checkpoint, old_store, cell.resolved_config)
                count = checkpoint["cursor"]["next_pair"]
                if (
                    count <= 0
                    or count != len(checkpoint["completed_ids"])
                    or len(set(checkpoint["completed_ids"])) != count
                ):
                    raise ValueError("Checkpoint membership is incomplete or duplicated")
            input_payload = old_store.verify(source_identity["parents"][0])
            verify_identity(
                input_payload["identity"], identities["inputs"], old_impl, stage="inputs"
            )
            imports.append((cell, identities, path, report, checkpoint, input_payload))
            rows.append(
                {
                    "arm": cell.arm_id,
                    "action": (
                        "continue_checkpoint" if checkpoint is not None else "continue_unfinished"
                    ),
                    "completed_pairs": count,
                    "checkpoint_artifact": source_identity["artifact_id"],
                    "outputs": len(checkpoint["outputs"]) if checkpoint is not None else 0,
                }
            )
    if not verify_only:
        for cell, identities, path, report, saved, extra in imports:
            index = runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json")
            if index.exists():
                # A resumed worker must preserve its newer completed/checkpoint state.
                continue
            if cell.arm_id == "brief_256_binary":
                for stage in STAGES:
                    outputs = {
                        name: old_store._blob(v["sha256"])
                        for name, v in saved[stage]["outputs"].items()
                    }
                    if stage == "extraction":
                        outputs["stats/execution_measurement.json"] = (
                            json.dumps(extra, indent=2, sort_keys=True) + "\n"
                        ).encode()
                    store.publish(identities[stage], outputs)
                artifacts = {stage: identities[stage]["artifact_id"] for stage in STAGES}
            else:
                store.import_artifact(old_root, extra["identity"]["artifact_id"])
                if (
                    saved is not None
                    and store.latest_checkpoint(identities["extraction"]["artifact_id"]) is None
                ):
                    store.checkpoint(
                        identities["extraction"],
                        completed_ids=saved["completed_ids"],
                        cursor=saved["cursor"],
                        outputs={
                            name: old_store._blob(v["sha256"])
                            for name, v in saved["outputs"].items()
                        },
                        state={
                            **saved["state"],
                            "migration": "e07-ledger-retry-v1",
                            "source_artifact": saved["identity"]["artifact_id"],
                        },
                    )
                artifacts = {"inputs": identities["inputs"]["artifact_id"]}
            write(
                index,
                {
                    "artifacts": artifacts,
                    "attempt_id": report["recovery"]["attempt_id"],
                    "output_dir": str(path.parent),
                    "migration": "e07-ledger-retry-v1",
                    "source_manifest": binding(path),
                },
                immutable=True,
            )
    result = {
        "status": "pass",
        "verify_only": verify_only,
        "rows": rows,
        "repair_record": settings["repair_record"],
    }
    write(
        runtime / ("e07-ledger-verification.json" if verify_only else "e07-ledger-imports.json"),
        result,
    )
    return result
