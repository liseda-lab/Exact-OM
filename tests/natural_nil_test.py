import json
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.impl.models.selector.nil_head import (
    fit_nil_artifact,
    nil_metrics,
    remove_development_positives,
    source_decision_records,
)
from tests.grouped_fitting_test import TinyDataset, TinyScorer, selector, tiny_runner


def task():
    rows, labels, reference = [], [], set()
    for kind, score in (("in_pool", 0.9), ("ontology_nil", 0.1), ("pool_miss", 0.45)):
        for index in range(3):
            source = f"{kind}-{index}"
            labels.append({"Src": source, "Status": kind})
            for rank in range(2):
                rows.append(
                    {
                        "Src": source,
                        "Tgt": f"{source}-{rank}",
                        "S_base": score - rank * 0.05,
                        "S_final": score - rank * 0.05,
                    }
                )
            if kind != "ontology_nil":
                reference.add((source, f"{source}-0" if kind == "in_pool" else "outside"))
    labels.append({"Src": "unresolved", "Status": "unknown"})
    return pd.DataFrame(rows), pd.DataFrame(labels), reference


def test_natural_nil_fit_is_grouped_and_excludes_unknowns(tmp_path, monkeypatch):
    import exact.impl.models.selector.nil_head as module

    frame, labels, reference = task()
    application = {
        "dataset_signature": "report",
        "source_ids": ["report-source"],
        "nil_label_semantics": "natural",
    }
    path = tmp_path / "nil.json"
    artifact = fit_nil_artifact(frame, labels, reference, path, application=application)
    assert artifact["unknown_labels_excluded"] == 1
    assert "unresolved" not in artifact["training_sources"]
    assert len(artifact["oof_predictions"]) == 9
    assert all(
        not set(fold["training_sources"]) & set(fold["heldout_sources"])
        for fold in artifact["folds"]
    )
    path.unlink()
    original = module._fit
    calls = []

    def fitted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(module, "_fit", fitted)
    assert fit_nil_artifact(frame, labels, reference, path, application=application) == artifact
    assert len(calls) == 1
    with pytest.raises(ValueError, match="independently annotated"):
        fit_nil_artifact(
            frame,
            labels,
            reference,
            tmp_path / "unsafe.json",
            application={**application, "nil_label_semantics": "unknown"},
        )
    with pytest.raises(ValueError, match="conflicts"):
        fit_nil_artifact(
            frame,
            labels,
            reference | {("ontology_nil-0", "positive")},
            tmp_path / "bad.json",
            application=application,
        )


def test_empty_pool_and_displayed_none_do_not_establish_natural_nil():
    frame = pd.DataFrame(
        {
            "Src": ["none"],
            "Tgt": ["candidate"],
            "selection_abstained": [True],
            "llm_source_choice": ["__NONE__"],
        }
    )
    records = source_decision_records(frame, ["none", "empty"])
    assert all(
        row["absence_semantics"] == "unknown" and row["ontology_nil_probability"] is None
        for row in records
    )
    assert next(row for row in records if row["Src"] == "empty")["candidate_count"] == 0


def test_pool_miss_diagnostic_and_nil_metrics_keep_absence_types_separate():
    frame, labels, reference = task()
    stripped, manifest = remove_development_positives(
        frame, reference, role="development", negative_label_policy="complete_reference"
    )
    assert not set(zip(stripped.Src, stripped.Tgt)) & reference
    assert not manifest["ontology_nil_claim"]
    with pytest.raises(ValueError, match="development"):
        remove_development_positives(
            frame, reference, role="final", negative_label_policy="complete_reference"
        )
    gold = pd.DataFrame(
        {"Src": ["nil", "miss", "unknown"], "Status": ["ontology_nil", "pool_miss", "unknown"]}
    )
    result = nil_metrics(
        [{"Src": source, "absence_semantics": "ontology_nil"} for source in gold.Src],
        gold,
        {("miss", "outside")},
        [],
    )
    assert result["natural_nil"] == {"TP": 1, "FP": 1, "FN": 0, "F1": pytest.approx(2 / 3)}
    assert result["pool_miss_as_nil"] == 1 and result["unknown_sources_excluded"] == 1


def test_training_pool_natural_nil_consumer_keeps_explicit_nil_sources(tmp_path):
    frame, labels, reference = task()
    pool = tmp_path / "pool.tsv"
    frame[["Src", "Tgt"]].to_csv(pool, sep="\t", index=False)
    reference_path = tmp_path / "reference.tsv"
    pd.DataFrame(sorted(reference), columns=["Src", "Tgt"]).to_csv(
        reference_path, sep="\t", index=False
    )
    label_path = tmp_path / "source-labels.tsv"
    labels.to_csv(label_path, sep="\t", index=False)
    head = selector()
    head.nil_config.update(
        mode="fitted", training_source_labels=str(label_path), label_semantics="natural"
    )
    data = TinyDataset(pd.DataFrame({"Src": ["report"], "Tgt": ["candidate"]}))
    runner = tiny_runner(tmp_path, data, TinyScorer(), head)
    runner.training_candidates_file_path = pool
    runner.training_reference_file_path = reference_path
    runner.supervision_config = {"negative_label_policy": "unknown"}
    runner.fit_training_pool(batch_size=4)
    artifact = json.loads(open(head.nil_config["artifact"]).read())
    assert set(artifact["training_sources"]) == set(labels[labels.Status != "unknown"].Src)
    report = frame.iloc[:2].copy()
    report["Src"] = "report"
    report["P_rank"] = [0.7, 0.3]
    report["S_select"] = [0.8, 0.0]
    result = head._apply_joint_nil_ranking(
        report, dataset=SimpleNamespace(dataset_signature=data.dataset_signature)
    )
    assert result.Q_match.sum() + result.Q_nil.iloc[0] + result.Q_pool_miss.iloc[
        0
    ] == pytest.approx(1.0)
