"""Resume E23 rich-100 across a reviewed training-storage retention-only change."""

from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

STAGES = ("inputs", "extraction", "evaluation")
_ALLOWED = {"exact/impl/trainer/fitting.py"}
_MIGRATION = "training-storage-retention-v1"


def verify_code(old_code, code, repair):
    from exact.experiments.runtime import _code_identity

    changes = repair.get("changes", {})
    if (
        repair.get("migration") != _MIGRATION
        or repair.get("scientific_choices_unchanged") is not True
        or repair.get("reporting_labels_exposed") is not False
        or set(changes) != _ALLOWED
    ):
        raise ValueError("Unreviewed training storage migration")
    old_scopes = []
    for evaluation in (False, True):
        before, after = (_code_identity(root, evaluation=evaluation) for root in (old_code, code))
        old, new = dict(before["files"]), dict(after["files"])
        for name, change in changes.items():
            if name not in old and name not in new:
                continue
            if old.pop(name, None) != change["before"] or new.pop(name, None) != change["after"]:
                raise ValueError("Training storage repair code hashes differ")
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


def verify_raw_shards(snapshot, source_cell):
    """Verify bounded original shards, complete source membership and graph sidecars."""
    from exact.impl.models.graph_head import verify_graph_manifests
    from exact.utils.fitted_artifacts import fingerprint

    identity = snapshot.get("training_identity", "")
    if snapshot.get("schema_version") != 1 or not re.fullmatch("[a-f0-9]{64}", identity):
        raise ValueError("Invalid retained training identity")
    outputs, sources, manifests, row_count = {}, set(), None, 0
    for item in snapshot.get("files", []):
        path = verified(item["file"])
        if (
            path.parent != source_cell / "fitting" / identity
            or not re.fullmatch("[a-f0-9]{64}\\.json", path.name)
            or path.stat().st_size > 256 * 1024**2
        ):
            raise ValueError("Raw training shard is outside its reviewed identity or bound")
        relative = f"fitting/{identity}/{path.name}"
        if relative in outputs:
            raise ValueError("Repeated retained training shard")
        payload = read(path)
        members, rows = payload["source_ids"], payload["rows"]
        pairs = [(str(row["Src"]), str(row["Tgt"])) for row in rows]
        if (
            members != item["source_ids"]
            or len(rows) != item["row_count"]
            or not members
            or len(set(members)) != len(members)
            or {source for source, _ in pairs} != set(members)
            or len(set(pairs)) != len(pairs)
            or sources & set(members)
            or fingerprint(members) != path.stem
        ):
            raise ValueError("Retained training shard source/pair membership differs")
        for row in rows:
            current = row["graph_features"]["graph_fingerprints"]
            if manifests is None:
                manifests = current
            elif current != manifests:
                raise ValueError("Retained training rows mix graph fingerprints")
        outputs[relative] = path
        sources.update(members)
        row_count += len(rows)
    if (
        not sources
        or manifests is None
        or len(sources) != snapshot["source_count"]
        or row_count != snapshot["row_count"]
    ):
        raise ValueError("Retained training snapshot is incomplete")
    expected = {manifests[side]["manifest_sha256"] for side in ("src", "tgt")}
    seen = set()
    for item in snapshot["graph_manifests"]:
        sidecar = verified(item)
        if (
            sidecar.parent != source_cell / "fitting/graph-manifests"
            or sidecar.stem not in expected
            or sidecar.stem in seen
        ):
            raise ValueError("Retained graph sidecar binding differs")
        seen.add(sidecar.stem)
        outputs["fitting/graph-manifests/" + sidecar.name] = sidecar
    if seen != expected:
        raise ValueError("Retained graph provenance is incomplete")
    verify_graph_manifests(manifests, source_cell / "fitting/graph-manifests")
    return outputs, len(sources)


def verify_terminal(settings):
    completion = read(verified(settings["source_completion"]))
    wrapper_exit = verified(settings["source_wrapper_exit"]).read_text().strip()
    launch = read(verified(settings["source_launch"]))
    if (
        completion.get("status") != "failed"
        or completion.get("exit_code") != 1
        or wrapper_exit != "75"
        or completion.get("step_id") != launch.get("step_id")
        or completion.get("dispatch_nonce") != launch.get("dispatch_nonce")
    ):
        raise ValueError("Source must match its terminal storage-guard failure receipts")


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
        recipe["scientific_step"] != "E23-rich-100"
        or len(cells) != 2
        or {c.arm_id for c in cells} != {"rich_100_off", "rich_100_inductive"}
        or any(
            c.task_id != "D1-global_alignment" or c.seed != 17 or c.source_cap != 300 for c in cells
        )
    ):
        raise ValueError("Storage recovery requires both unchanged E23 rich-100 arms")
    return suite, cells


def import_saved(recipe, campaign, runtime, code, *, verify_only=False):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _hash, _verify_materialized

    settings = recipe["training_retention_repair"]
    verify_terminal(settings)
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
        if cell.arm_id == "rich_100_off":
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
            original_identity = read(identity_path)["identity"]
            ArtifactStore._identity(original_identity)
            verify_identity(
                original_identity, identities["extraction"], old_impl, stage="extraction"
            )
            if (
                report.get("return_code") != -15
                or old_store.latest_checkpoint(original_identity["artifact_id"]) is not None
            ):
                raise ValueError("Expected stopped treatment with raw features only")
            original_inputs = old_store.verify(original_identity["parents"][0])
            verify_identity(
                original_inputs["identity"], identities["inputs"], old_impl, stage="inputs"
            )
            outputs, count = verify_raw_shards(
                read(verified(settings["raw_features"])), path.parent
            )
            imports.append((cell, recovery, path, report, outputs, None))
            rows.append(
                {
                    "arm": cell.arm_id,
                    "action": "continue_retained_raw_features",
                    "training_sources": count,
                }
            )
    if not verify_only:
        for cell, recovery, path, report, saved, measurement in imports:
            identities = recovery.identities
            if cell.arm_id == "rich_100_inductive":
                if store.latest_checkpoint(identities["extraction"]["artifact_id"]) is None:
                    store.publish(
                        identities["inputs"],
                        recovery.input_files or {"_locked_inputs/manifest.json": b"{}"},
                    )
                    store.checkpoint(
                        identities["extraction"],
                        completed_ids=[],
                        cursor={"stage": "raw-training-features", "fitting_complete": False},
                        outputs=saved,
                        state={"migration": _MIGRATION, "raw_features": settings["raw_features"]},
                    )
                continue
            index = runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json")
            if index.exists():
                retained = read(index)
                if retained.get("artifacts") != {
                    stage: identities[stage]["artifact_id"] for stage in STAGES
                }:
                    raise ValueError("Retained complete control index differs")
                for identity in identities.values():
                    store.verify(identity["artifact_id"])
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
        runtime / ("retention-verification.json" if verify_only else "retention-imports.json"),
        result,
    )
    return result
