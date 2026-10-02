"""Fail-closed E14 resume boundaries; no ontology, model or hosted execution."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments.budget import BudgetLedger
from exact.experiments.recovery import stage_identity
from tools import recover_e14_ledger as recovery
from tools.prepared_batch import binding, read, write


def test_only_reviewed_ledger_code_can_change(tmp_path):
    old, new = (tmp_path / name for name in ("old", "new"))
    for root in (old, new):
        for path in ("exact/llm/ledger.py", "exact/llm/routing.py", "exact/models.py"):
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("original")
    for path in recovery.FILES:
        (new / path).write_text("synchronization-only")
    repair = dict(
        migration=recovery.MIGRATION,
        scientific_choices_unchanged=True,
        reporting_labels_exposed=False,
        changes={
            name: dict(before=binding(old / name)["sha256"], after=binding(new / name)["sha256"])
            for name in recovery.FILES
        },
    )
    recovery.verify_code(old, new, repair)
    (new / "exact/models.py").write_text("scientific change")
    with pytest.raises(ValueError, match="outside reviewed"):
        recovery.verify_code(old, new, repair)


def test_identity_retains_every_numerical_input():
    old_impl, new_impl = {"files": {"ledger": "old"}}, {"files": {"ledger": "new"}}
    args = dict(
        stage="extraction",
        parameters={"threshold": 0.4},
        inputs={"train": "a" * 64},
        parents=[],
        role="development",
        entity_kind="class",
        seed=17,
        dependencies={"torch": "same"},
    )
    old = stage_identity(**args, implementation=old_impl)
    new = stage_identity(**args, implementation=new_impl)
    recovery.verify_identity(old, new, old_impl, stage="extraction")
    for key, value in (
        ("seed", 18),
        ("role", "final"),
        ("inputs", {"train": "different"}),
        ("parameters", {"threshold": 0.5}),
        ("dependencies", {"torch": "new"}),
    ):
        changed = copy.deepcopy(new)
        changed[key] = value
        with pytest.raises(ValueError, match="beyond hosted"):
            recovery.verify_identity(old, changed, old_impl, stage="extraction")


def _checkpoint(tmp_path, count):
    dataset = tmp_path / "dataset"
    dataset.write_text(
        "Src,SrcKind,Tgt,TgtKind,inference\n"
        + "".join(f"S{i},class,T{i},class,True\n" for i in range(5844))
    )
    saved = tmp_path / "inference"
    write(
        saved,
        dict(
            kind="inference",
            total_examples=5844,
            processed_examples=count,
            mappings_count=count,
            results_json_count=count,
            explanation_records_count=count,
            explanation_index_path="../explanations/index.json",
        ),
    )
    cp = dict(
        cursor={"next_pair": count, "dataset_rows": 5844},
        completed_ids=[json.dumps([f"S{i}", "class", f"T{i}", "class"]) for i in range(count)],
        outputs={
            "dataset/dataset.csv": {"sha256": "dataset"},
            "checkpoints/inference_1.json": {"sha256": "inference"},
        },
    )
    return cp, SimpleNamespace(_blob=lambda digest: tmp_path / digest)


@pytest.mark.parametrize("count", [1024, 2048])
def test_checkpoint_keeps_exact_prefix_and_runner_cursor(tmp_path, count):
    cp, store = _checkpoint(tmp_path, count)
    recovery.verify_checkpoint(cp, store, count)
    cp["completed_ids"][0], cp["completed_ids"][1] = cp["completed_ids"][1], cp["completed_ids"][0]
    with pytest.raises(ValueError, match="ordered dataset prefix"):
        recovery.verify_checkpoint(cp, store, count)
    cp["completed_ids"][0], cp["completed_ids"][1] = cp["completed_ids"][1], cp["completed_ids"][0]
    saved = read(tmp_path / "inference")
    saved["processed_examples"] -= 1
    write(tmp_path / "inference", saved)
    with pytest.raises(ValueError, match="runner checkpoint"):
        recovery.verify_checkpoint(cp, store, count)


def _account(tmp_path):
    limits = dict(
        requests_cap=100, tokens_cap=1000, node_hours_cap=10, envelopes_hours={"reserve": 10}
    )
    source = BudgetLedger(tmp_path / "source.json", limits)
    source.admit("old", group="reserve", seconds=0, forecast_known=False)
    source.finish("old", start=1, end=2, status="complete", requests=1, tokens=10, actual_usd=None)
    path = tmp_path / "plan.json"
    write(
        path,
        dict(
            evidence=binding(tmp_path / "source.json"),
            start=2,
            end=5,
            seconds=3,
            requests=0,
            tokens=0,
            work_id="recovery-accounting/E14-inventory-recovery-01/pre-screen",
        ),
    )
    write(tmp_path / "destination.json", source.snapshot())
    return {"e14_ledger_repair": {"accounting_reconciliation": binding(path)}}, BudgetLedger(
        tmp_path / "destination.json", limits
    )


def test_missing_setup_charge_is_exactly_once(tmp_path):
    recipe, ledger = _account(tmp_path)
    recovery.charge_missing_setup(recipe, ledger)
    first = ledger.snapshot()
    recovery.charge_missing_setup(recipe, ledger)
    assert ledger.snapshot() == first
    correction = first["work"]["recovery-accounting/E14-inventory-recovery-01/pre-screen"]
    assert (correction["seconds"], correction["requests"], correction["tokens"]) == (3, 0, 0)
    assert first["work"]["old"]["actual_usd"] is None


def test_missing_setup_charge_rejects_overlap(tmp_path):
    recipe, ledger = _account(tmp_path)
    ledger.admit("already-accounted", group="reserve", seconds=0, forecast_known=False)
    ledger.finish(
        "already-accounted", start=3, end=4, status="complete", requests=0, tokens=0, actual_usd=0
    )
    with pytest.raises(ValueError, match="overlaps"):
        recovery.charge_missing_setup(recipe, ledger)


def test_missing_setup_charge_rejects_lost_history(tmp_path):
    recipe, ledger = _account(tmp_path)
    path = tmp_path / "destination.json"
    state = read(path)
    state["work"]["old"]["tokens"] = 9
    write(path, state)
    with pytest.raises(ValueError, match="loses original"):
        recovery.charge_missing_setup(recipe, ledger)


def _fitted(tmp_path):
    import collections
    import random

    from exact.impl.models.selector.fitting import fingerprint

    training = [dict(Src=f"S{i}", Tgt=f"T{i}", Relation=("=", "<", ">")[i % 3]) for i in range(6)]
    training_path = tmp_path / "train.tsv"
    training_path.write_text(
        "Src\tTgt\tRelation\n"
        + "".join(f"{r['Src']}\t{r['Tgt']}\t{r['Relation']}\n" for r in training)
    )
    provenance = dict(
        recipe="multinomial_l2_0.01_v1",
        seed=17,
        training=training,
        application={"source_ids": [f"application-{i}" for i in range(300)]},
        features="same",
        profiles={"same": True},
    )
    identity = fingerprint(provenance)
    head = tmp_path / "cell/fitting/relation_head.json"
    artifact = dict(
        fit_identity=identity,
        fit_provenance=provenance,
        feature_schema=["a"],
        weights=[[0, 0, 0]],
        bias=[0, 0, 0],
        relation_counts=dict(collections.Counter(r["Relation"] for r in training)),
        folds=[],
        oof_predictions=[],
    )
    groups = sorted(r["Src"] for r in training)
    random.Random(17).shuffle(groups)
    bindings = []
    for i in range(3):
        held = groups[i::3]
        fold = dict(heldout_sources=held, training_sources=sorted(set(groups) - set(held)))
        artifact["folds"].append(fold)
        predictions = [
            dict(
                Src=r["Src"],
                Tgt=r["Tgt"],
                true_relation=r["Relation"],
                probabilities=[0.2, 0.3, 0.5],
            )
            for r in training
            if r["Src"] in held
        ]
        artifact["oof_predictions"].extend(predictions)
        path = head.parent / (head.name + ".folds") / f"{identity}-{i}.json"
        write(path, {**fold, "predictions": predictions})
        bindings.append(binding(path))
    write(head, artifact)
    row = dict(
        head=binding(head),
        training_input=binding(training_path),
        fit_identity=identity,
        training_rows=6,
        training_sources=6,
        folds_verified=bindings,
    )
    return row, head.parent.parent, {"matching": {"relation_training_file": str(training_path)}}


def test_fitted_head_keeps_bound_training_and_all_folds(tmp_path):
    row, cell, config = _fitted(tmp_path)
    outputs = recovery.verify_fitted(row, cell, config)
    assert set(outputs) == {
        "fitting/relation_head.json",
        *(str(Path(v["path"]).relative_to(cell)) for v in row["folds_verified"]),
    }
    path = Path(row["folds_verified"][0]["path"])
    fold = read(path)
    fold["heldout_sources"] = ["outside-training"]
    write(path, fold)
    row["folds_verified"][0] = binding(path)
    with pytest.raises(ValueError, match="fold membership"):
        recovery.verify_fitted(row, cell, config)


def test_fitted_head_rejects_changed_training_labels_even_with_new_binding(tmp_path):
    row, cell, config = _fitted(tmp_path)
    path = Path(row["training_input"]["path"])
    path.write_text(path.read_text().replace("S0\tT0\t=", "S0\tT0\t<"))
    row["training_input"] = binding(path)
    with pytest.raises(ValueError, match="training labels"):
        recovery.verify_fitted(row, cell, config)
