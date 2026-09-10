import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.experiments.nil_evaluation import evaluate_source_labels
from exact.utils.provenance import sha256_file


def evaluation_cell(tmp_path):
    labels = tmp_path / "evaluation-labels.tsv"
    pd.DataFrame(
        {
            "Src": ["mapped", "nil", "miss", "unknown"],
            "Status": ["mapped", "ontology_nil", "mapped", "unknown"],
        }
    ).to_csv(labels, sep="\t", index=False)
    references = tmp_path / "valid.tsv"
    pd.DataFrame({"SrcEntity": ["mapped", "miss"], "TgtEntity": ["t", "outside"]}).to_csv(
        references, sep="\t", index=False
    )
    records = [
        {
            "Src": "mapped",
            "candidates": [{"target": "t", "S_final": 0.9}, {"target": "other", "S_final": 0.1}],
            "emitted_targets": ["t"],
            "absence_semantics": "in_pool",
        },
        {
            "Src": "nil",
            "candidates": [],
            "emitted_targets": [],
            "absence_semantics": "ontology_nil",
        },
        {
            "Src": "miss",
            "candidates": [{"target": "wrong"}],
            "emitted_targets": [],
            "absence_semantics": "ontology_nil",
        },
        {"Src": "unknown", "candidates": [], "emitted_targets": []},
        {"Src": "unlabeled", "candidates": [], "emitted_targets": []},
    ]
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "source_universe": [row["Src"] for row in records],
        "records": records,
    }
    (tmp_path / "source_decisions.json").write_text(json.dumps(trace))
    return SimpleNamespace(
        output_dir=tmp_path,
        split_role="development",
        reference_role="valid",
        reference_completeness="complete",
        diagnostics={
            "role": "development",
            "reference_role": "valid",
            "evaluation_source_labels": {"path": str(labels), "sha256": sha256_file(labels)},
        },
        resolved_config={"data": {"root": str(tmp_path), "refs": {"valid": str(references)}}},
    )


def test_runtime_nil_metrics_use_bound_source_labels_and_keep_unknown_sources_out(tmp_path):
    cell = evaluation_cell(tmp_path)
    before = json.loads(json.dumps(cell.resolved_config))
    result = evaluate_source_labels(cell)
    assert cell.resolved_config == before
    assert "evaluation-labels.tsv" not in json.dumps(cell.resolved_config)
    assert result["metrics"]["labeled_sources"] == 3
    assert result["metrics"]["unknown_sources_excluded"] == 2
    assert result["metrics"]["pool_miss_as_nil"] == 1
    assert result["metrics"]["non_nil_MRR"] == 0.5
    assert result["metrics"]["non_nil_sources"] == 2
    assert result["metrics"]["nil_aware"]["F1"] == pytest.approx(2 / 3)
    assert result["metrics"]["natural_nil"] == {
        "TP": 1,
        "FP": 1,
        "FN": 0,
        "F1": pytest.approx(2 / 3),
    }
    report = json.loads(Path(result["artifact"]["path"]).read_text())
    assert report["evaluation_only"] and len(report["source_universe"]) == 5
    assert evaluate_source_labels(cell) == result


def test_incomplete_pair_reference_does_not_make_unknown_mapping_a_false_positive(tmp_path):
    cell = evaluation_cell(tmp_path)
    cell.reference_completeness = "known_incomplete"
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    trace["records"][0]["emitted_targets"] = ["other"]
    path.write_text(json.dumps(trace))
    metrics = evaluate_source_labels(cell)["metrics"]
    assert metrics["nil_aware"]["status"] == "unavailable"
    assert metrics["nil_aware"]["unassessed_emitted_pairs"] == 1
    assert metrics["natural_nil"]["F1"] == pytest.approx(2 / 3)


def test_wrong_role_and_training_label_reuse_fail_before_reading_labels(tmp_path):
    cell = evaluation_cell(tmp_path)
    labels = Path(cell.diagnostics["evaluation_source_labels"]["path"])
    labels.unlink()
    cell.diagnostics["role"] = "reporting"
    with pytest.raises(ValueError, match="role differs"):
        evaluate_source_labels(cell)
    cell.diagnostics["role"] = "development"
    cell.resolved_config["matching"] = {"nil": {"training_source_labels": str(labels)}}
    with pytest.raises(ValueError, match="training_source_labels"):
        evaluate_source_labels(cell)


def test_changed_labels_and_missing_final_sources_fail_closed(tmp_path):
    cell = evaluation_cell(tmp_path)
    binding = cell.diagnostics["evaluation_source_labels"]
    Path(binding["path"]).write_text("Src\tStatus\nnil\tunknown\n")
    with pytest.raises(ValueError, match="changed after binding"):
        evaluate_source_labels(cell)
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    trace["records"].pop()
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="one final decision"):
        evaluate_source_labels(cell)


def test_incomplete_reference_does_not_turn_known_mapped_source_into_pool_miss(tmp_path):
    cell = evaluation_cell(tmp_path)
    cell.reference_completeness = "known_incomplete"
    result = evaluate_source_labels(cell)
    assert result["metrics"]["pool_miss_as_nil"] == 0
    report = json.loads(Path(result["artifact"]["path"]).read_text())
    assert report["pool_status_unresolved_sources"] == ["miss"]
    labels = Path(cell.diagnostics["evaluation_source_labels"]["path"])
    frame = pd.read_csv(labels, sep="\t")
    frame.loc[frame.Src == "miss", "Status"] = "pool_miss"
    frame.to_csv(labels, sep="\t", index=False)
    cell.diagnostics["evaluation_source_labels"]["sha256"] = sha256_file(labels)
    assert evaluate_source_labels(cell)["metrics"]["pool_miss_as_nil"] == 1


def test_non_nil_mrr_uses_joint_ranks_and_keeps_missed_sources_in_denominator(tmp_path):
    cell = evaluation_cell(tmp_path)
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    candidates = trace["records"][0]["candidates"]
    candidates[0]["candidate_joint_rank"] = 2
    candidates[1]["candidate_joint_rank"] = 3
    path.write_text(json.dumps(trace))
    assert evaluate_source_labels(cell)["metrics"]["non_nil_MRR"] == 0.25
    for candidate in candidates:
        candidate.pop("candidate_joint_rank")
    candidates[0]["Q_match"], candidates[1]["Q_match"] = 0.3, 0.1
    candidates[0]["Q_nil"] = candidates[1]["Q_nil"] = 0.6
    path.write_text(json.dumps(trace))
    assert evaluate_source_labels(cell)["metrics"]["non_nil_MRR"] == 0.25
