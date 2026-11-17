"""Benchmark rejection labels are complete only within the annotated candidate pool."""


import json


from pathlib import Path


from types import SimpleNamespace


import pandas as pd


import pytest


from exact.impl.models.selector.fitting import safe_training_labels


from exact.impl.models.selector.nil_head import (
    fit_nil_artifact,
    source_decision_records,
)


from exact.utils.provenance import sha256_file


from tests.grouped_fitting_test import TinyDataset, TinyScorer, selector, tiny_runner


from tests.natural_nil_test import task


from tests.nil_evaluation_test import evaluation_cell


def benchmark_task():
    frame, labels, reference = task()
    frame = frame[~frame.Src.str.startswith("pool_miss")].copy()
    labels = labels[labels.Status != "pool_miss"].copy()
    labels.Status = labels.Status.replace({"ontology_nil": "benchmark_nil"})
    reference = {pair for pair in reference if not pair[0].startswith("pool_miss")}
    frame["confirmed_label"] = [int(pair in reference) for pair in zip(frame.Src, frame.Tgt)]
    return frame, labels, reference


def test_benchmark_source_head_fits_actual_training_pool_without_ontology_nil_claim(tmp_path):
    frame, labels, reference = benchmark_task()
    pool, refs, source_labels = (
        tmp_path / name for name in ("pool.tsv", "refs.tsv", "sources.tsv")
    )
    frame[["Src", "Tgt", "confirmed_label"]].to_csv(pool, sep="\t", index=False)
    pd.DataFrame(sorted(reference), columns=["Src", "Tgt"]).to_csv(refs, sep="\t", index=False)
    labels.to_csv(source_labels, sep="\t", index=False)
    head = selector()
    head.nil_config.update(
        mode="fitted", training_source_labels=str(source_labels), label_semantics="benchmark_pool"
    )
    data = TinyDataset(pd.DataFrame({"Src": ["report"], "Tgt": ["candidate"]}))
    runner = tiny_runner(tmp_path, data, TinyScorer(), head)
    runner.training_candidates_file_path, runner.training_reference_file_path = pool, refs
    runner.supervision_config = {"negative_label_policy": "confirmed_negatives"}
    runner.fit_training_pool(batch_size=4)
    artifact = json.loads(Path(head.nil_config["artifact"]).read_text())
    assert artifact["kind"] == "benchmark_nil_head" and artifact["ontology_nil_claim"] is False
    assert artifact["statuses"] == ["in_pool", "benchmark_nil", "pool_miss"]
    assert artifact["source_label_counts"] == {"in_pool": 3, "benchmark_nil": 3}
    report = frame.iloc[:2].copy()
    report["Src"], report["P_rank"], report["S_select"] = "report", [0.7, 0.3], [0.8, 0.0]
    decision = source_decision_records(report, ["report"], artifact=artifact)[0]
    assert decision["ontology_nil_probability"] is None
    assert decision["benchmark_nil_probability"] is not None
    result = head._apply_joint_nil_ranking(
        report, dataset=SimpleNamespace(dataset_signature=data.dataset_signature)
    )
    assert result.Q_match.sum() + result.Q_nil.iloc[0] + result.Q_pool_miss.iloc[
        0
    ] == pytest.approx(1)


def benchmark_cell(tmp_path):
    cell = evaluation_cell(tmp_path)
    cell.reference_completeness = "known_incomplete"
    binding = cell.diagnostics["evaluation_source_labels"]
    labels = pd.read_csv(binding["path"], sep="\t")
    labels.Status = labels.Status.replace({"ontology_nil": "benchmark_nil"})
    labels.loc[labels.Src == "miss", "Status"] = "unknown"
    labels.to_csv(binding["path"], sep="\t", index=False)
    binding["sha256"] = sha256_file(binding["path"])
    reference = Path(cell.resolved_config["data"]["refs"]["valid"])
    reference.write_text("SrcEntity\tTgtEntity\nmapped\tt\n")
    candidates = tmp_path / "evaluation-candidate-labels.tsv"
    candidates.write_text("Src\tTgt\tconfirmed_label\nmapped\tt\t1\nmapped\tother\t0\nnil\tx\t0\n")
    cell.diagnostics.update(
        label_semantics="benchmark_pool",
        evaluation_candidate_labels={"path": str(candidates), "sha256": sha256_file(candidates)},
    )
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    trace["records"][1].update(
        candidates=[{"target": "x", "S_final": 0.1}], absence_semantics="unknown"
    )
    trace["records"][2]["candidates"] = []
    path.write_text(json.dumps(trace))
    return cell
