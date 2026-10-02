"""Recover E23 from verified compact raw features after graph provenance deduplication."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

STAGES = ("inputs", "extraction", "evaluation")
_ALLOWED = {
    "exact/impl/models/graph_head.py",
    "exact/impl/models/pair_adaptive_scorer.py",
    "exact/impl/trainer/fitting.py",
    "exact/utils/fitted_artifacts.py",
    "exact/llm/ledger.py",
    "exact/llm/routing.py",
}
_MIGRATION = "graph-provenance-storage-v1"


def verify_code(old_code, code, repair):
    from exact.experiments.runtime import _code_identity

    changes = repair.get("changes", {})
    if (
        repair.get("migration") != _MIGRATION
        or repair.get("scientific_choices_unchanged") is not True
        or repair.get("reporting_labels_exposed") is not False
        or not {"exact/impl/models/graph_head.py"}.issubset(changes)
        or not set(changes) <= _ALLOWED
    ):
        raise ValueError("Unreviewed graph storage migration")
    old_scopes = []
    for evaluation in (False, True):
        before, after = (_code_identity(root, evaluation=evaluation) for root in (old_code, code))
        old, new = dict(before["files"]), dict(after["files"])
        for name, change in changes.items():
            if name not in old and name not in new:
                continue
            if old.pop(name, None) != change["before"] or new.pop(name, None) != change["after"]:
                raise ValueError("Graph storage repair code hashes differ")
        if old != new:
            raise ValueError("Implementation differs outside the reviewed storage repair")
        old_scopes.append(before)
    # Raw checkpoint directory identity must retain exactly the same construction.
    for name, function in [
        ("exact/impl/trainer/fitting.py", "fit_training_pool"),
        ("exact/impl/models/pair_adaptive_scorer.py", "_runtime_fingerprint_payload"),
    ]:
        definitions = []
        for root in (old_code, code):
            node = next(
                n
                for n in ast.walk(ast.parse((root / name).read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == function
            )
            if function == "fit_training_pool":
                # Start at raw file parsing, excluding unrelated disabled consumer guards.
                start = next(
                    i
                    for i, n in enumerate(node.body)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "raw" for t in n.targets)
                )
                end = next(
                    i
                    for i, n in enumerate(node.body)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "identity" for t in n.targets)
                )
                node = ast.Module(body=node.body[start : end + 1], type_ignores=[])
            definitions.append(ast.dump(node, include_attributes=False))
        if definitions[0] != definitions[1]:
            raise ValueError("Raw training identity construction changed")
    return tuple(old_scopes)


def verify_identity(saved, expected, old_impl, *, stage, old_extraction=None, old_evaluation=None):
    target = copy.deepcopy(expected)
    target.pop("artifact_id")
    if stage == "extraction":
        target["implementation"] = old_impl
    elif stage == "evaluation":
        if old_evaluation is None:
            raise ValueError("Evaluation migration requires its reviewed original implementation")
        target["implementation"] = old_evaluation
        target["parents"] = [old_extraction]
    source = dict(saved)
    source.pop("artifact_id")
    if source != target:
        raise ValueError("Graph recovery input/config/role/seed identity differs: " + stage)


def verify_compacted_shards(snapshot, source_cell):
    """Bind every retained value to the streaming compaction receipt and full manifests."""
    from exact.impl.models.graph_head import verify_graph_manifests
    from exact.utils.fitted_artifacts import fingerprint
    from tools.compact_graph_features import _preserved

    identity = snapshot.get("training_identity", "")
    if snapshot.get("schema_version") != 1 or not re.fullmatch("[a-f0-9]{64}", identity):
        raise ValueError("Invalid compact raw training identity")
    outputs, sources, manifests = {}, set(), None
    for item in snapshot.get("files", []):
        path, receipt_path = verified(item["file"]), verified(item["receipt"])
        receipt = read(receipt_path)
        if (
            path.parent != source_cell / "fitting" / identity
            or not re.fullmatch("[a-f0-9]{64}\\.json", path.name)
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "compacted"
            or receipt.get("path") != str(path)
            or receipt.get("compact_sha256") != item["file"]["sha256"]
            or receipt.get("compact_bytes") != path.stat().st_size
            or not re.fullmatch("[a-f0-9]{64}", receipt.get("original_sha256", ""))
        ):
            raise ValueError("Compacted shard is outside its verified receipt or identity")
        relative = f"fitting/{identity}/{path.name}"
        if relative in outputs:
            raise ValueError("Repeated compact shard")
        payload = read(path)
        members, rows = payload["source_ids"], payload["rows"]
        pairs = [(str(row["Src"]), str(row["Tgt"])) for row in rows]
        if (
            members != receipt["source_ids"]
            or len(rows) != receipt["row_count"]
            or not members
            or len(set(members)) != len(members)
            or {source for source, _ in pairs} != set(members)
            or len(set(pairs)) != len(pairs)
            or sources & set(members)
            or fingerprint(members) != path.stem
        ):
            raise ValueError("Compacted shard membership differs")
        digest = hashlib.sha256()
        for row in rows:
            digest.update(_preserved(row).encode())
            current = row["graph_features"]["graph_fingerprints"]
            if manifests is None:
                manifests = current
            elif current != manifests:
                raise ValueError("Compacted training rows mix graph fingerprints")
        if digest.hexdigest() != receipt["preserved_rows_sha256"]:
            raise ValueError("Compaction receipt does not match preserved scientific values")
        for item_binding in receipt["graph_manifests"]:
            sidecar = verified(
                {"path": item_binding["path"], "sha256": item_binding["file_sha256"]}
            )
            if (
                sidecar.parent != source_cell / "fitting/graph-manifests"
                or sidecar.stem != item_binding["manifest_sha256"]
                or current[item_binding["side"]]["manifest_sha256"] != sidecar.stem
            ):
                raise ValueError("Compacted graph sidecar binding differs")
            outputs["fitting/graph-manifests/" + sidecar.name] = sidecar
        outputs[relative] = path
        sources.update(members)
    if not sources or manifests is None:
        raise ValueError("Storage recovery has no complete raw feature shards")
    verify_graph_manifests(manifests, source_cell / "fitting/graph-manifests")
    return outputs, len(sources)


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
            raise ValueError("Storage recovery requires verified completed producers")
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
        recipe["scientific_step"] != "E23-rich-50"
        or len(cells) != 2
        or {c.arm_id for c in cells} != {"rich_50_off", "rich_50_inductive"}
    ):
        raise ValueError("Storage recovery requires both unchanged E23 rich-50 arms")
    return suite, cells


def import_saved(recipe, campaign, runtime, code, *, verify_only=False):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _hash, _verify_materialized

    settings = recipe["graph_storage_repair"]
    old_root = Path(settings["source_runtime"])
    old_impl, old_evaluation = verify_code(
        Path(settings["source_code"]), code, read(verified(settings["repair_record"]))
    )
    suite, cells = _cells(recipe, campaign, runtime)
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
            raise ValueError("Storage source cell is outside its declared runtime")
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
            raise ValueError("Storage recovery configuration or scientific population differs")
        recovery = CellRecovery(
            replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True}),
            harness._provenance_payload(cell, suite, workdir=code),
            code,
        )
        identities = recovery.identities
        if cell.arm_id == "rich_50_off":
            if (
                report.get("status") != "complete"
                or report.get("return_code") != 0
                or report.get("extraction_complete") is not True
            ):
                raise ValueError("Reusable storage control is incomplete")
            artifacts, saved = report["recovery"]["artifacts"], {}
            if set(artifacts) != set(STAGES):
                raise ValueError("Control lacks complete artifact provenance")
            for stage in STAGES:
                payload = old_store.verify(artifacts[stage])
                verify_identity(
                    payload["identity"],
                    identities[stage],
                    old_impl,
                    stage=stage,
                    old_extraction=artifacts["extraction"],
                    old_evaluation=old_evaluation,
                )
                if stage != "inputs":
                    _verify_materialized(old_store, payload, path.parent)
                saved[stage] = payload
            item = saved["extraction"]["outputs"]["stats/execution_measurement.json"]
            measurement = read(old_store._blob(item["sha256"]))
            if (
                measurement.get("artifact_id") != artifacts["extraction"]
                or measurement.get("schema_version") != 1
            ):
                raise ValueError("Storage control execution measurement mismatch")
            measurement = {
                **measurement,
                "artifact_id": identities["extraction"]["artifact_id"],
                "identity_migration": {
                    "reason": _MIGRATION,
                    "source_artifact_id": artifacts["extraction"],
                    "source_measurement_sha256": item["sha256"],
                    "previous": measurement.get("identity_migration"),
                },
            }
            imports.append((cell, recovery, path, report, saved, measurement))
            rows.append({"arm": cell.arm_id, "action": "reuse_complete"})
        else:
            if report.get("status") != "failed" or report.get("extraction_complete") is not False:
                raise ValueError("Expected incomplete quota-failed treatment")
            identity_path = verified(settings["source_runtime_binding"])
            if identity_path != path.parent / "recovery-runtime.json":
                raise ValueError("Raw feature runtime binding differs")
            verify_identity(
                read(identity_path)["identity"],
                identities["extraction"],
                old_impl,
                stage="extraction",
            )
            outputs, count = verify_compacted_shards(
                read(verified(settings["raw_features"])), path.parent
            )
            imports.append((cell, recovery, path, report, outputs, None))
            rows.append(
                {
                    "arm": cell.arm_id,
                    "action": "continue_compact_raw_features",
                    "training_sources": count,
                }
            )
    if not verify_only:
        for cell, recovery, path, report, saved, measurement in imports:
            identities = recovery.identities
            if cell.arm_id == "rich_50_inductive":
                if store.latest_checkpoint(identities["extraction"]["artifact_id"]) is None:
                    store.publish(
                        identities["inputs"],
                        recovery.input_files or {"_locked_inputs/manifest.json": b"{}"},
                    )
                    store.checkpoint(
                        identities["extraction"],
                        completed_ids=sorted(saved),
                        cursor={"stage": "raw-training-features", "fitting_complete": False},
                        outputs=saved,
                        state={"migration": _MIGRATION, "raw_features": settings["raw_features"]},
                    )
                continue
            index = runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json")
            if index.exists():
                continue
            for stage in STAGES:
                outputs = {
                    name: old_store._blob(item["sha256"])
                    for name, item in saved[stage]["outputs"].items()
                }
                if stage == "extraction":
                    outputs["stats/execution_measurement.json"] = (
                        json.dumps(measurement, sort_keys=True, indent=2) + "\n"
                    ).encode()
                store.publish(identities[stage], outputs)
            write(
                index,
                {
                    "artifacts": {stage: identities[stage]["artifact_id"] for stage in STAGES},
                    "attempt_id": report["recovery"]["attempt_id"],
                    "output_dir": str(path.parent),
                    "migration": _MIGRATION,
                    "source_manifest": binding(path),
                },
                immutable=True,
            )
    result = {
        "status": "pass",
        "verify_only": verify_only,
        "rows": rows,
        "repair_record": settings["repair_record"],
        "raw_features": settings["raw_features"],
        "original_charges_retained": True,
        "fitted_treatment_artifacts_imported": 0,
    }
    write(
        runtime
        / ("graph-storage-verification.json" if verify_only else "graph-storage-imports.json"),
        result,
    )
    return result
