"""Restore E14 only across reviewed hosted-ledger synchronization/authorization changes."""

from __future__ import annotations

import collections
import copy
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

FILES = {"exact/llm/ledger.py", "exact/llm/routing.py"}
ARMS = {
    "all_equivalent",
    "hierarchy_heuristic",
    "graph_entailment",
    "learned_three_way",
    "semantic_then_learned",
}
COUNTS = {"hierarchy_heuristic": 1024, "graph_entailment": 2048}
LEARNED = {"learned_three_way", "semantic_then_learned"}
MIGRATION = "e14-hosted-ledger-recovery-v1"


def verify_code(old_code, code, repair):
    from exact.experiments.runtime import _code_identity

    old = _code_identity(old_code, evaluation=False)
    new = _code_identity(code, evaluation=False)
    changes = repair.get("changes", {})
    if repair.get("migration") != MIGRATION or set(changes) != FILES:
        raise ValueError("Only the pinned hosted synchronization repair may migrate E14")
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


def verify_checkpoint(checkpoint, store, expected_count):
    """Verify the exact saved ordered prefix, runner cursor and explanation coverage."""
    outputs = checkpoint["outputs"]
    cursor = checkpoint["cursor"]
    ids = checkpoint["completed_ids"]
    if (
        cursor != {"next_pair": expected_count, "dataset_rows": 5844}
        or len(set(ids)) != expected_count
    ):
        raise ValueError("E14 checkpoint cursor or unique membership differs")
    with store._blob(outputs["dataset/dataset.csv"]["sha256"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # The durable CSV also retains exact/prefiltered pairs, while the runner's
    # active frame contains only inference=True rows in the same order.
    if any(row.get("inference") not in {"True", "False"} for row in rows):
        raise ValueError("E14 checkpoint dataset lacks explicit inference membership")
    ordered = [
        json.dumps([row[key] for key in ("Src", "SrcKind", "Tgt", "TgtKind")])
        for row in rows
        if row["inference"] == "True"
    ]
    if len(ordered) != cursor["dataset_rows"] or ids != ordered[:expected_count]:
        raise ValueError("E14 checkpoint is not the exact ordered dataset prefix")
    names = [name for name in outputs if name.startswith("checkpoints/inference_")]
    if len(names) != 1:
        raise ValueError("E14 checkpoint must bind one inference cursor")
    saved = read(store._blob(outputs[names[0]]["sha256"]))
    required = dict(
        kind="inference",
        total_examples=5844,
        processed_examples=expected_count,
        mappings_count=expected_count,
        results_json_count=expected_count,
        explanation_records_count=expected_count,
        explanation_index_path="../explanations/index.json",
    )
    if any(saved.get(key) != value for key, value in required.items()):
        raise ValueError("E14 runner checkpoint cursor or explanation coverage differs")


def verify_fitted(row, cell, config):
    """Recheck bound training-only heads/folds; native feature identity is checked at reuse."""
    from exact.impl.models.selector.fitting import fingerprint
    from exact.io.relation_head import typed_reference_frame

    head = verified(row["head"])
    if head != cell / "fitting/relation_head.json":
        raise ValueError("E14 fitted head is outside its original cell")
    training_path = verified(row["training_input"])
    if training_path != Path(config["matching"]["relation_training_file"]):
        raise ValueError("E14 fitted head training input differs")
    artifact = read(head)
    provenance = artifact["fit_provenance"]
    training = {(r["Src"], r["Tgt"], r["Relation"]) for r in provenance["training"]}
    declared = set(
        typed_reference_frame(training_path)[["Src", "Tgt", "Relation"]].itertuples(
            index=False, name=None
        )
    )
    groups = {r["Src"] for r in provenance["training"]}
    application = set(provenance["application"]["source_ids"])
    if (
        fingerprint(provenance) != artifact["fit_identity"]
        or artifact["fit_identity"] != row["fit_identity"]
        or provenance["seed"] != 17
        or provenance["recipe"] != "multinomial_l2_0.01_v1"
        or training != declared
        or len(training) != len(provenance["training"])
        or len(training) != row["training_rows"]
        or len(groups) != row["training_sources"]
        or len(application) != 300
        or groups & application
        or dict(collections.Counter(r["Relation"] for r in provenance["training"]))
        != artifact["relation_counts"]
    ):
        raise ValueError("E14 fitted head provenance, training labels or population differs")
    ordered = sorted(groups)
    random.Random(17).shuffle(ordered)
    outputs, seen, predictions = {"fitting/relation_head.json": head}, set(), []
    if len(row["folds_verified"]) != 3 or len(artifact["folds"]) != 3:
        raise ValueError("E14 fitted head lacks its three source-disjoint folds")
    for fold, bound in enumerate(row["folds_verified"]):
        path = verified(bound)
        if path != head.parent / (head.name + ".folds") / f"{artifact['fit_identity']}-{fold}.json":
            raise ValueError("E14 fitted fold path differs")
        saved = read(path)
        held = set(saved["heldout_sources"])
        if (
            held != set(ordered[fold::3])
            or set(saved["training_sources"]) != groups - held
            or seen & held
            or {k: v for k, v in saved.items() if k != "predictions"} != artifact["folds"][fold]
            or {(p["Src"], p["Tgt"], p["true_relation"]) for p in saved["predictions"]}
            != {r for r in training if r[0] in held}
        ):
            raise ValueError("E14 fitted OOF fold membership differs")
        for prediction in saved["predictions"]:
            probs = prediction["probabilities"]
            if (
                len(probs) != 3
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in probs)
                or abs(sum(probs) - 1) >= 1e-8
            ):
                raise ValueError("E14 fitted OOF probabilities are invalid")
        seen.update(held)
        predictions.extend(saved["predictions"])
        outputs[str(path.relative_to(cell))] = path
    if (
        seen != groups
        or predictions != artifact["oof_predictions"]
        or len(artifact["weights"]) != len(artifact["feature_schema"])
        or any(len(row) != 3 for row in artifact["weights"])
        or len(artifact["bias"]) != 3
    ):
        raise ValueError("E14 fitted head or complete OOF record differs")
    return outputs


def cells_for(recipe, campaign, runtime):
    from exact.experiments import harness
    from exact.experiments.campaign import materialize_campaign

    suite = materialize_campaign(campaign, runtime / "declarations/screen", stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == "E14")
    cells = harness.build_cells(
        suite, source, stage="screen", output_root=runtime.parent, inherited_overlay={}
    )
    if (
        recipe["scientific_step"] != "E14"
        or source.config.depends_on
        or len(cells) != 5
        or {c.arm_id for c in cells} != ARMS
        or any(
            c.task_id != "T0-global_alignment" or c.seed != 17 or c.source_cap != 300 for c in cells
        )
    ):
        raise ValueError("E14 recovery requires its five unchanged original cells")
    return suite, cells


def import_saved(recipe, campaign, runtime, code, *, verify_only=False):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _hash

    settings = recipe["e14_ledger_repair"]
    old_root = Path(settings["source_runtime"])
    old_impl = verify_code(
        Path(settings["source_code"]), code, read(verified(settings["repair_record"]))
    )
    terminal = read(verified(settings["source_completion"]))
    if terminal.get("status") != "failed" or terminal.get("exit_code") != 1:
        raise ValueError("E14 source must have a verified terminal failure")
    audit = read(verified(settings["artifact_verification"]))
    fitted = read(verified(settings["fitted_verification"]))
    saved_rows = {row["arm"]: row for row in audit["rows"]}
    fitted_rows = {row["arm"]: row for row in fitted["rows"]}
    if (
        audit["status"] != "pass"
        or fitted["status"] != "pass"
        or set(saved_rows) != ARMS
        or set(fitted_rows) != LEARNED
    ):
        raise ValueError("E14 saved artifact audit is incomplete")
    suite, cells = cells_for(recipe, campaign, runtime)
    old_store, store = ArtifactStore(old_root), ArtifactStore(runtime)
    imports, rows = [], []
    for cell in cells:
        row = saved_rows[cell.arm_id]
        path = verified(row["manifest"])
        if (
            path
            != old_root
            / "screen/runs/E14"
            / cell.arm_id
            / cell.task_id
            / f"seed-{cell.seed}"
            / "experiment_manifest.json"
        ):
            raise ValueError("E14 source cell is outside the declared runtime")
        report = read(path)
        expected = dict(
            experiment_id="E14",
            arm_id=cell.arm_id,
            task_id=cell.task_id,
            stage="screen",
            seed=17,
            source_cap=300,
            split_role=cell.split_role,
            reference_role=cell.reference_role,
            arm_role=cell.arm_role,
            resolved_config_hash=cell.config_hash,
            generate_rationales=False,
            resolved_supervision=cell.resolved_supervision,
            status="failed",
            extraction_complete=False,
            candidate_pool_fingerprint=row["candidate_pool_fingerprint"],
        )
        config = load_yaml_mapping(verified(row["config"]))
        if any(report.get(k) != v for k, v in expected.items()) or config != cell.resolved_config:
            raise ValueError("E14 original config, role, population, pool or seed differs")
        recovery = CellRecovery(
            replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True}),
            harness._provenance_payload(cell, suite, workdir=code),
            code,
        )
        identities = recovery.identities
        source_identity = read(path.parent / "recovery-runtime.json")["identity"]
        ArtifactStore._identity(source_identity)
        if source_identity["artifact_id"] != row["extraction_identity"]:
            raise ValueError("E14 original extraction identity differs from audit")
        verify_identity(source_identity, identities["extraction"], old_impl, stage="extraction")
        inputs = old_store.verify(source_identity["parents"][0])
        verify_identity(inputs["identity"], identities["inputs"], old_impl, stage="inputs")
        checkpoint = old_store.latest_checkpoint(source_identity["artifact_id"])
        outputs, count = {}, COUNTS.get(cell.arm_id, 0)
        if count:
            pinned = read(verified(row["checkpoint"]))
            if checkpoint != pinned or checkpoint["identity"] != source_identity:
                raise ValueError("E14 original checkpoint differs from reviewed boundary")
            verify_checkpoint(checkpoint, old_store, count)
            outputs = {
                name: old_store._blob(item["sha256"])
                for name, item in checkpoint["outputs"].items()
            }
        elif checkpoint is not None or row["completed_pairs"] != 0:
            raise ValueError("Unexpected E14 extraction checkpoint")
        if cell.arm_id in LEARNED:
            outputs.update(verify_fitted(fitted_rows[cell.arm_id], path.parent, config))
        imports.append((cell, identities, path, report, inputs, checkpoint, outputs))
        rows.append(
            dict(
                arm=cell.arm_id,
                completed_pairs=count,
                outputs=len(outputs),
                fitted_head=cell.arm_id in LEARNED,
                source_identity=source_identity["artifact_id"],
                target_identity=identities["extraction"]["artifact_id"],
            )
        )
    # Every arm/byte/identity has passed before publishing any imported checkpoint.
    if not verify_only:
        for cell, identities, path, report, inputs, checkpoint, outputs in imports:
            index = runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json")
            if index.exists():
                retained = read(index)
                artifacts = retained.get("artifacts", {})
                if not artifacts or any(
                    stage not in identities or identifier != identities[stage]["artifact_id"]
                    for stage, identifier in artifacts.items()
                ):
                    raise ValueError("Existing E14 recovery index has incompatible identity")
                for identifier in artifacts.values():
                    store.verify(identifier)
                continue  # Preserve later worker checkpoints/results on an explicit restart.
            store.import_artifact(old_root, inputs["identity"]["artifact_id"])
            if outputs and store.latest_checkpoint(identities["extraction"]["artifact_id"]) is None:
                store.checkpoint(
                    identities["extraction"],
                    completed_ids=checkpoint["completed_ids"] if checkpoint else [],
                    cursor=(
                        checkpoint["cursor"]
                        if checkpoint
                        else {"stage": "relation-fit", "next_pair": 0}
                    ),
                    outputs=outputs,
                    state={
                        "migration": MIGRATION,
                        "source_manifest": binding(path),
                        "source_checkpoint": saved_rows[cell.arm_id].get("checkpoint"),
                    },
                )
            write(
                index,
                dict(
                    artifacts={"inputs": identities["inputs"]["artifact_id"]},
                    attempt_id=report["recovery"]["attempt_id"],
                    output_dir=str(path.parent),
                    migration=MIGRATION,
                    source_manifest=binding(path),
                ),
                immutable=True,
            )
    result = dict(
        status="pass",
        verify_only=verify_only,
        rows=rows,
        repair_record=settings["repair_record"],
        completed_cells_imported=0,
        original_charges_retained=True,
    )
    write(
        runtime / ("e14-ledger-verification.json" if verify_only else "e14-ledger-imports.json"),
        result,
    )
    return result


def charge_missing_setup(recipe, ledger):
    """Retain the audited pre-screen interval once, only in the imported destination account."""
    plan = read(verified(recipe["e14_ledger_repair"]["accounting_reconciliation"]))
    evidence = read(verified(plan["evidence"]))
    state = ledger.snapshot()
    if any(state["work"].get(key) != value for key, value in evidence["work"].items()):
        raise ValueError("E14 recovery account loses original cumulative work")
    if (
        plan["work_id"] != "recovery-accounting/E14-inventory-recovery-01/pre-screen"
        or plan["seconds"] != plan["end"] - plan["start"]
        or plan["requests"] != 0
        or plan["tokens"] != 0
    ):
        raise ValueError("E14 missing setup accounting evidence differs")
    expected = dict(
        start=plan["start"], end=plan["end"], status="failed", requests=0, tokens=0, actual_usd=0
    )
    if plan["work_id"] in state["work"]:
        if any(state["work"][plan["work_id"]].get(k) != v for k, v in expected.items()):
            raise ValueError("E14 existing setup correction differs")
        return
    if any(
        min(row.get("end", plan["start"]), plan["end"])
        > max(row.get("start", plan["end"]), plan["start"])
        for row in state["work"].values()
    ):
        raise ValueError("E14 setup correction overlaps previously charged work")
    ledger.admit(plan["work_id"], group="reserve", seconds=0, forecast_known=False)
    ledger.finish(plan["work_id"], **expected)
