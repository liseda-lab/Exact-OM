"""Budget repair reuses raw evidence only and rejects unreviewed numerical changes."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments.recovery import ArtifactStore, stage_identity
from exact.utils.fitted_artifacts import fingerprint
from tools import recover_e22_labels as repair
from tools.prepared_batch import binding, write


def config():
    return {
        "seed": 17,
        "data": {"source": "ontology", "train_candidates": "train-only"},
        "matching": {"fusion": {"mode": "analytic_shipped"}},
        "llm": {"experiment": {"gate": {"mode": "off"}}},
        "pipeline": [{"name": "Scorer", "params": {"tau": 0.5}}, {"name": "CandidateSetSelector"}],
        "selector": {"rerank": {"mode": "pairwise"}},
        "supervision": {
            "label_budget": 25,
            "components": {"rerank": "supervised", "accept": "supervised", "fusion": "label_free"},
        },
    }


def test_only_downstream_selector_controls_are_removed():
    old, new = config(), config()
    new["supervision"]["label_budget"] = 100
    new["selector"]["rerank"]["mode"] = "gam"
    assert repair.feature_config(old) == repair.feature_config(new)
    new["pipeline"][0]["params"]["tau"] = 0.9
    assert repair.feature_config(old) != repair.feature_config(new)
    new = config()
    new["data"]["train_candidates"] = "different-training-pool"
    assert repair.feature_config(old) != repair.feature_config(new)


@pytest.mark.parametrize("change", ["fusion", "graph", "gate", "transfer", "llm"])
def test_feature_reuse_rejects_fitted_scorers(change):
    value = config()
    if change == "fusion":
        value["matching"]["fusion"]["mode"] = "analytic_fitted"
    elif change == "graph":
        value["matching"]["channels"] = {"graph": {"mode": "inductive"}}
    elif change == "gate":
        value["llm"]["experiment"]["gate"]["mode"] = "learned"
    elif change == "transfer":
        value["supervision"]["transfer_artifact"] = "donor.json"
    else:
        value["supervision"]["components"]["llm"] = "supervised"
    with pytest.raises(ValueError, match="unchanged unfitted"):
        repair.feature_config(value)


def snapshot(root, *, members=None, rows=None):
    source = root / "screen/runs/E22-policy/fixed_count/D1-global_alignment/seed-17"
    identity = "b" * 64
    source_config = source / "_inputs/resolved.config.yaml"
    write(source_config, config())
    members = members or ["s1", "s2"]
    rows = rows or [{"Src": source, "Tgt": "t"} for source in members]
    path = source / "fitting" / identity / (fingerprint(members) + ".json")
    write(path, {"source_ids": members, "rows": rows})
    return {
        "schema_version": 1,
        "training_identity": identity,
        "source_config": binding(source_config),
        "files": [{"name": path.name, **binding(path)}],
    }


def test_partial_raw_shards_are_reusable_without_fitted_artifacts(tmp_path):
    sample = snapshot(tmp_path)
    value, files = repair.verify_raw_snapshot(sample, tmp_path)
    assert value == config()
    assert len(files) == 1
    assert files[0][0] == Path(sample["files"][0]["path"]).name


@pytest.mark.parametrize(
    "damage",
    ["checksum", "membership", "duplicate", "fit", "path", "temporary", "identity", "overlap"],
)
def test_raw_snapshot_rejects_corruption_or_nonraw_artifacts(tmp_path, damage):
    sample = snapshot(tmp_path)
    item = sample["files"][0]
    path = Path(item["path"])
    if damage == "checksum":
        path.write_text("damaged")
    elif damage == "membership":
        sample = snapshot(tmp_path, rows=[{"Src": "other", "Tgt": "t"}])
    elif damage == "duplicate":
        sample["files"].append(dict(item))
    elif damage == "fit":
        item["name"] = "selector.json"
    elif damage == "path":
        destination = path.parent.parent / path.name
        destination.write_bytes(path.read_bytes())
        item.update(binding(destination))
    elif damage == "temporary":
        item["name"] += ".partial"
    elif damage == "identity":
        sample["training_identity"] = "wrong"
    else:
        second = snapshot(tmp_path, members=["s2", "s3"])
        sample["files"].extend(second["files"])
    with pytest.raises(ValueError):
        repair.verify_raw_snapshot(sample, tmp_path)


def test_reviewed_code_change_cannot_hide_raw_feature_change(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    text = "start\n        raw = read_table(Path(path))\n        feature = 1\n        from exact.impl.models.selector.label_budget import select_label_budget\nold\n"
    for root in (old, new):
        path = root / repair.FITTING
        path.parent.mkdir(parents=True)
        path.write_text(text)
    (new / repair.FITTING).write_text(text.replace("\nold\n", "\nnew\n"))
    record = {
        "schema_version": 1,
        "changes": {
            repair.FITTING: {
                "before": binding(old / repair.FITTING)["sha256"],
                "after": binding(new / repair.FITTING)["sha256"],
            }
        },
    }
    assert repair.verify_code_change(old, new, record)
    (new / repair.FITTING).write_text(text.replace("feature = 1", "feature = 2"))
    record["changes"][repair.FITTING]["after"] = binding(new / repair.FITTING)["sha256"]
    with pytest.raises(ValueError, match="feature generation changed"):
        repair.verify_code_change(old, new, record)


def label_free(root):
    source = root / "screen/runs/E22/label_free/D0-global_alignment/seed-17"
    source.mkdir(parents=True)
    cfg = config()
    cfg["supervision"]["label_budget"] = None
    cfg["supervision"]["components"] = {
        name: "label_free" for name in cfg["supervision"]["components"]
    }
    write(source / "_inputs/resolved.config.yaml", cfg)
    cell = SimpleNamespace(
        arm_id="label_free",
        resolved_config=cfg,
        config_hash="config",
        task_id="D0-global_alignment",
        seed=17,
        split_role="development",
        reference_role="valid",
        source_cap=300,
        arm_role="baseline",
    )
    old_impl = {"sha256": "old"}
    store, identities, saved = ArtifactStore(root), {}, {}
    for stage in repair.STAGES:
        old = stage_identity(
            stage,
            parameters={},
            inputs={},
            implementation=old_impl if stage == "extraction" else {"schema": "unchanged"},
            dependencies={},
            role="development",
            entity_kind="class",
            seed=17,
            parents=[saved["extraction"]["artifact_id"]] if stage == "evaluation" else [],
        )
        new = stage_identity(
            stage,
            parameters={},
            inputs={},
            implementation={"sha256": "new"} if stage == "extraction" else {"schema": "unchanged"},
            dependencies={},
            role="development",
            entity_kind="class",
            seed=17,
            parents=[identities["extraction"]["artifact_id"]] if stage == "evaluation" else [],
        )
        identities[stage], saved[stage] = new, old
        outputs = {"predictions.csv": b"source,target\ns,t\n"}
        if stage == "extraction":
            outputs["stats/execution_measurement.json"] = json.dumps(
                {
                    "schema_version": 1,
                    "artifact_id": old["artifact_id"],
                    "wall_seconds": 321,
                    "origin_attempt": "first-run",
                    "identity_migration": {"reason": "earlier-key-repair"},
                }
            ).encode()
        if stage == "evaluation":
            outputs["evaluation/evaluation_results.json"] = json.dumps(
                {
                    "builtin": {"F1": 0.718},
                    "meta": {
                        "refs": {
                            "alignment": {
                                "path": "/original/predictions.csv",
                                "sha256": "same-predictions",
                                "rows": 293,
                            }
                        }
                    },
                }
            ).encode()
        store.publish(old, outputs)
        if stage != "inputs":
            store.restore(old["artifact_id"], source)
    report = dict(
        experiment_id="E22",
        arm_id="label_free",
        stage="screen",
        status="complete",
        return_code=0,
        extraction_complete=True,
        generate_rationales=False,
        task_id=cell.task_id,
        seed=17,
        split_role="development",
        reference_role="valid",
        source_cap=300,
        arm_role="baseline",
        resolved_config_hash="config",
        recovery={
            "artifacts": {stage: identity["artifact_id"] for stage, identity in saved.items()},
            "attempt_id": "original",
        },
    )
    path = source / "experiment_manifest.json"
    write(path, report)
    return cell, SimpleNamespace(identities=identities), path, store, old_impl


def test_label_free_rebinding_preserves_original_measurement_and_prior_migration(tmp_path):
    args = label_free(tmp_path)
    _, _, measurement = repair.verify_label_free(*args)
    result = json.loads(measurement)
    assert result["wall_seconds"] == 321
    assert result["origin_attempt"] == "first-run"
    assert result["artifact_id"] == args[1].identities["extraction"]["artifact_id"]
    assert result["identity_migration"]["previous"] == {"reason": "earlier-key-repair"}


@pytest.mark.parametrize(
    "damage", ["supervised", "fit", "seed", "exposed", "implementation", "config"]
)
def test_only_unchanged_unfitted_label_free_control_may_migrate(tmp_path, damage):
    args = list(label_free(tmp_path))
    if damage == "supervised":
        args[0].arm_id = "budget_25"
    elif damage == "fit":
        (args[2].parent / "fitting").mkdir()
    elif damage == "seed":
        args[0].seed = 29
    elif damage == "exposed":
        (args[2].parent / "predictions.csv").write_text("changed")
    elif damage == "implementation":
        args[4] = {"sha256": "unreviewed"}
    else:
        args[0].resolved_config = copy.deepcopy(args[0].resolved_config)
        args[0].resolved_config["data"]["source"] = "changed"
    with pytest.raises(ValueError):
        repair.verify_label_free(*args)


def test_policy_count_rule_can_change_without_changing_raw_features():
    source = {
        "inputs": {
            "references/train": "labels",
            "train_candidates": "pairs",
            "artifact/supervision.artifacts.training_units": "groups",
        },
        "dependencies": {"torch": "frozen"},
        "role": "development",
        "entity_kind": "class",
        "seed": 17,
    }
    target = copy.deepcopy(source)
    target["inputs"]["artifact/supervision.auto_policy.artifact"] = "new-fitted-count-rule"
    repair.verify_raw_identity(source, target)
    for key in (
        "references/train",
        "train_candidates",
        "artifact/supervision.artifacts.training_units",
    ):
        damaged = copy.deepcopy(target)
        damaged["inputs"][key] = "changed"
        with pytest.raises(ValueError, match="Raw feature inputs"):
            repair.verify_raw_identity(source, damaged)


def test_inactive_analytic_gate_is_compatible_but_negative_policy_is_retained():
    old = config()
    old["llm"]["experiment"].update(enabled=False, gate={"mode": "analytic"})
    new = copy.deepcopy(old)
    assert repair.feature_config(old) == repair.feature_config(new)
    new["supervision"]["negative_label_policy"] = "complete_reference"
    assert repair.feature_config(old) != repair.feature_config(new)


@pytest.mark.parametrize("damage", [None, "metric", "checksum", "rows"])
def test_label_free_allows_only_portable_evaluation_provenance(tmp_path, damage):
    args = label_free(tmp_path)
    report_path = args[2].parent / "evaluation/evaluation_results.json"
    report = json.loads(report_path.read_text())
    report["meta"]["refs"]["alignment"]["path"] = "/restored/predictions.csv"
    if damage == "metric":
        report["builtin"]["F1"] = 0.9
    elif damage == "checksum":
        report["meta"]["refs"]["alignment"]["sha256"] = "changed-predictions"
    elif damage == "rows":
        report["meta"]["refs"]["alignment"]["rows"] = 999
    write(report_path, report)
    if damage:
        with pytest.raises(ValueError, match="changed consumed output"):
            repair.verify_label_free(*args)
    else:
        repair.verify_label_free(*args)
