"""Benchmark rejection labels are complete only within the annotated candidate pool."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.experiments.nil_evaluation import evaluate_source_labels
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


def test_complete_benchmark_pool_metrics_do_not_require_complete_ontology_reference(tmp_path):
    cell = benchmark_cell(tmp_path)
    result = evaluate_source_labels(cell)
    assert result["metrics"]["nil_aware"]["F1"] == 1
    assert result["metrics"]["benchmark_nil"]["F1"] == 1
    assert result["metrics"]["non_nil_MRR"] == 1
    assert result["metrics"]["unknown_sources_excluded"] == 3
    assert "natural_nil" not in result["metrics"]
    report = json.loads(Path(result["artifact"]["path"]).read_text())
    assert report["ontology_nil_claim"] is False
    assert report["reference_completeness"] == "known_incomplete"
    # A confirmed wrong candidate is assessed even though the global reference is incomplete.
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    trace["records"][0]["emitted_targets"] = ["other"]
    path.write_text(json.dumps(trace))
    assert evaluate_source_labels(cell)["metrics"]["nil_aware"] == {
        "TP": 1,
        "FP": 1,
        "FN": 1,
        "F1": 0.5,
    }


def test_unknown_outside_benchmark_pool_never_becomes_confirmed_negative(tmp_path):
    cell = benchmark_cell(tmp_path)
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    trace["records"][1]["emitted_targets"] = ["unannotated-outside-target"]
    trace["records"][1]["candidates"].append(
        {"target": "unannotated-outside-target", "S_final": 0.9}
    )
    path.write_text(json.dumps(trace))
    result = evaluate_source_labels(cell)
    assert result["metrics"]["nil_aware"]["status"] == "unavailable"
    assert result["metrics"]["nil_aware"]["unassessed_emitted_pairs"] == 1
    assert result["metrics"]["non_nil_MRR"] is None
    assert result["metrics"]["benchmark_nil"]["status"] == "unavailable"
    assert result["metrics"]["applicability"]["benchmark_nil"] is False
    assert "FN" not in result["metrics"]["benchmark_nil"]
    frame, labels, reference = benchmark_task()
    frame.loc[len(frame)] = {"Src": "outside", "Tgt": "unknown", "confirmed_label": None}
    safe, _ = safe_training_labels(
        frame, reference, {"negative_label_policy": "confirmed_negatives"}
    )
    assert "outside" not in set(safe.Src)
    with pytest.raises(ValueError, match="complete binary"):
        fit_nil_artifact(
            frame,
            labels,
            reference,
            tmp_path / "bad.json",
            application={"nil_label_semantics": "benchmark_pool"},
        )


def test_campaign_binds_benchmark_semantics_and_never_inherits_d0_fitted_weights(
    tmp_path, monkeypatch
):
    from exact.experiments import harness
    from exact.experiments.campaign import load_campaign, materialize_campaign
    from exact.experiments.preparation import prepare_campaign
    from tests.campaign_preparation_test import _cases

    case = benchmark_cell(tmp_path)
    cases = _cases()
    root = Path(__file__).resolve().parents[1]
    universe = tmp_path / "universe.txt"
    universe.write_text("mapped\nnil\nunknown\nmiss\nunlabeled\n")
    pool = tmp_path / "gold-stripped-pool.tsv"
    pool.write_text("Src\tTgt\nmapped\tt\nmapped\tother\nnil\tx\n")
    source, target = tmp_path / "source.owl", tmp_path / "target.owl"
    source.write_text("source")
    target.write_text("target")

    def binding(path):
        return {"path": str(path), "sha256": sha256_file(path)}

    train_labels = tmp_path / "train-source-labels.tsv"
    train_labels.write_text("Src\tStatus\ntrain\tbenchmark_nil\n")
    train_ref = tmp_path / "train-reference.tsv"
    train_ref.write_text("Src\tTgt\ntrain-positive\tt\n")
    cases["N0"].update(
        source=binding(source),
        target=binding(target),
        source_universe=binding(universe),
        references={
            "valid": binding(Path(case.resolved_config["data"]["refs"]["valid"])),
            "train": binding(train_ref),
        },
        candidates={"valid": binding(pool)},
        frozen_global_candidates={"valid": binding(pool)},
        evaluation_label_semantics="benchmark_pool",
        evaluation_source_labels=case.diagnostics["evaluation_source_labels"],
        evaluation_candidate_labels=case.diagnostics["evaluation_candidate_labels"],
        reference_completeness="known_incomplete",
        negative_policy="confirmed_only",
        overlay={"matching": {"nil": {"training_source_labels": str(train_labels)}}},
    )
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": cases},
        tmp_path / "prepared",
    )
    suite = materialize_campaign(path, tmp_path / "materialized", stage="screen")
    declaration = suite.by_id["E04"]
    declaration.config.implementation.status = "ready"
    lock, _ = load_campaign(path)
    assert next(step for step in lock.steps if step.id == "E04").inherits == ["E05_initial"]
    cells = harness.build_cells(suite, declaration, stage="screen", output_root=tmp_path / "runs")
    fitted = next(cell for cell in cells if cell.arm_id == "nil_fitted")
    assert fitted.resolved_supervision["nil"]["resolved"] == "supervised"
    assert fitted.resolved_supervision["accept"]["resolved"] == "label_free"
    for cell in cells:
        assert cell.resolved_config["matching"]["nil"]["label_semantics"] == "benchmark_pool"
        assert cell.resolved_config["data"]["candidate_provenance"] == "benchmark_supplied"
        assert cell.reference_completeness == "known_incomplete"
        assert (
            cell.diagnostics["evaluation_candidate_labels"]
            == case.diagnostics["evaluation_candidate_labels"]
        )
        assert "evaluation-candidate-labels" not in json.dumps(cell.resolved_config)
    with pytest.raises(ValueError, match="inherited fitted artifacts"):
        harness.build_cells(
            suite,
            declaration,
            stage="screen",
            output_root=tmp_path / "unsafe",
            inherited_overlay={"matching": {"calibration": {"artifact": "/D0/fitted.json"}}},
        )
    # The selection-facing metric reader retains the scoped endpoint beside generic metrics.
    evaluate_source_labels(case)
    monkeypatch.setattr(harness, "extract_evaluation_metrics", lambda _: {"F1": 0.1})
    metrics = harness.cell_metrics(case.output_dir)
    assert metrics["nil.nil_aware.F1"] == metrics["nil.benchmark_nil.F1"] == 1
    assert "nil.natural_nil.F1" not in metrics
